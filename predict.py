import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from data.dataset import get_eval_transform
from model.clip import CLIP


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filenames = sorted(
            f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.root, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


def _parse_aliases(class_name):
    aliases = [part.strip() for part in class_name.split(",") if part.strip()]
    if not aliases:
        aliases = [class_name.strip()]
    return aliases


def build_prompt_specs(class_names, templates, class_prompt_variants):
    """
    Build per-template prompt pools and class-to-prompt index mappings.
    """
    specs = []
    for template in templates:
        prompts = []
        class_prompt_indices = []
        for class_name in class_names:
            aliases = _parse_aliases(class_name)
            aliases.extend(class_prompt_variants.get(class_name, []))
            # Preserve order while deduplicating aliases.
            dedup_aliases = list(dict.fromkeys(alias.strip() for alias in aliases if alias.strip()))
            indices = []
            for alias in dedup_aliases:
                indices.append(len(prompts))
                prompts.append(template.format(alias))
            class_prompt_indices.append(indices)
        specs.append(
            {
                "template": template,
                "prompts": prompts,
                "class_prompt_indices": class_prompt_indices,
            }
        )
    return specs


def _normalize_template_weights(num_templates, template_weights):
    if not template_weights:
        return torch.ones(num_templates, dtype=torch.float32) / float(num_templates)

    if len(template_weights) != num_templates:
        raise ValueError(
            "Number of template weights must match number of templates. "
            f"Got {len(template_weights)} weights for {num_templates} templates."
        )
    weights = torch.tensor(template_weights, dtype=torch.float32)
    weight_sum = weights.sum().item()
    if weight_sum <= 0:
        raise ValueError("Template weights must sum to a positive value.")
    return weights / weight_sum


def _center_zoom(images, ratio):
    """
    Deterministic center crop and resize back to original resolution.
    """
    if ratio <= 0 or ratio > 1:
        raise ValueError("Zoom ratio must be in (0, 1].")
    if ratio == 1:
        return images
    _, _, h, w = images.shape
    crop_h = max(1, int(round(h * ratio)))
    crop_w = max(1, int(round(w * ratio)))
    top = (h - crop_h) // 2
    left = (w - crop_w) // 2
    cropped = images[:, :, top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, size=(h, w), mode="bilinear", align_corners=False)


def build_tta_views(images, tta_modes):
    views = []
    for mode in tta_modes:
        if mode == "base":
            views.append(images)
        elif mode == "hflip":
            views.append(torch.flip(images, dims=[3]))
        elif mode == "center_zoom_90":
            views.append(_center_zoom(images, ratio=0.90))
        elif mode == "center_zoom_80":
            views.append(_center_zoom(images, ratio=0.80))
        else:
            raise ValueError(f"Unknown TTA mode: {mode}")
    return views


def compute_class_logits(model, images, prompt_specs, template_weights):
    """
    Compute class logits with class-aware prompt pooling and template weighting.
    """
    per_template_logits = []
    for spec in prompt_specs:
        logits = model.compute_similarity(images, spec["prompts"])
        class_logits = [
            logits[:, idxs].mean(dim=1) for idxs in spec["class_prompt_indices"]
        ]
        per_template_logits.append(torch.stack(class_logits, dim=1))
    stacked = torch.stack(per_template_logits, dim=0)  # [T, B, C]
    return (stacked * template_weights[:, None, None]).sum(dim=0)


def predict_with_recipe(
    model,
    dataloader,
    class_names,
    templates,
    template_weights,
    class_prompt_variants,
    tta_modes,
):
    prompt_specs = build_prompt_specs(
        class_names=class_names,
        templates=templates,
        class_prompt_variants=class_prompt_variants,
    )
    normalized_weights = _normalize_template_weights(
        num_templates=len(templates),
        template_weights=template_weights,
    ).to(next(model.parameters()).device)

    results = []
    top1_conf_sum = 0.0
    top1_margin_sum = 0.0
    total = 0
    with torch.no_grad():
        for images, filenames in tqdm(dataloader, desc="Predicting"):
            images = images.to(next(model.parameters()).device)
            view_logits = []
            for view in build_tta_views(images, tta_modes):
                logits = compute_class_logits(
                    model=model,
                    images=view,
                    prompt_specs=prompt_specs,
                    template_weights=normalized_weights,
                )
                view_logits.append(logits)
            ensemble_logits = torch.stack(view_logits, dim=0).mean(dim=0)
            probs = torch.softmax(ensemble_logits, dim=1)
            top10 = torch.topk(probs, k=10, dim=-1)
            top10_ids = top10.indices.cpu().tolist()
            top10_probs = top10.values.cpu()
            for fname, ids, confs in zip(filenames, top10_ids, top10_probs):
                top1_conf = float(confs[0].item())
                top2_conf = float(confs[1].item())
                top1_conf_sum += top1_conf
                top1_margin_sum += (top1_conf - top2_conf)
                total += 1
                results.append({"filename": fname, "top10_ids": ids})

    diagnostics = {
        "num_samples": total,
        "avg_top1_confidence": top1_conf_sum / max(1, total),
        "avg_top1_top2_margin": top1_margin_sum / max(1, total),
    }
    return results, diagnostics


def save_predictions(path, results):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def run_ablation_sweep(model, dataloader, class_names):
    os.makedirs(Config.PRED_ABLATION_OUTPUT_DIR, exist_ok=True)
    baseline_templates = [
        "a photo of {}.",
        "a centered photo of {}.",
        "a close-up photo of {}.",
        "a cropped photo of {}.",
    ]
    exps = [
        {
            "name": "exp_a_baseline",
            "templates": baseline_templates,
            "weights": [],
            "variants": {},
            "tta_modes": ["base"],
        },
        {
            "name": "exp_b_prompt_pack",
            "templates": Config.EVAL_TEXT_TEMPLATES,
            "weights": [],
            "variants": {},
            "tta_modes": ["base"],
        },
        {
            "name": "exp_c_class_aware",
            "templates": Config.EVAL_TEXT_TEMPLATES,
            "weights": [],
            "variants": Config.TESTSET_CLASS_PROMPT_VARIANTS,
            "tta_modes": ["base"],
        },
        {
            "name": "exp_d_plus_tta",
            "templates": Config.EVAL_TEXT_TEMPLATES,
            "weights": [],
            "variants": Config.TESTSET_CLASS_PROMPT_VARIANTS,
            "tta_modes": Config.PRED_TTA_MODES,
        },
        {
            "name": "exp_e_plus_weights",
            "templates": Config.EVAL_TEXT_TEMPLATES,
            "weights": Config.EVAL_TEMPLATE_WEIGHTS,
            "variants": Config.TESTSET_CLASS_PROMPT_VARIANTS,
            "tta_modes": Config.PRED_TTA_MODES,
        },
    ]

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "output_dir": Config.PRED_ABLATION_OUTPUT_DIR,
        "experiments": [],
    }
    best_name = None
    best_score = None
    for exp in exps:
        print(f"Running ablation: {exp['name']}")
        results, diagnostics = predict_with_recipe(
            model=model,
            dataloader=dataloader,
            class_names=class_names,
            templates=exp["templates"],
            template_weights=exp["weights"],
            class_prompt_variants=exp["variants"],
            tta_modes=exp["tta_modes"],
        )
        output_path = os.path.join(Config.PRED_ABLATION_OUTPUT_DIR, f"{exp['name']}.json")
        save_predictions(output_path, results)
        record = {
            "name": exp["name"],
            "prediction_path": output_path,
            "templates": exp["templates"],
            "template_weights": exp["weights"],
            "tta_modes": exp["tta_modes"],
            "diagnostics": diagnostics,
        }
        summary["experiments"].append(record)
        score = diagnostics["avg_top1_top2_margin"]
        if best_score is None or score > best_score:
            best_score = score
            best_name = exp["name"]

    summary["best_experiment_by_margin_proxy"] = best_name
    summary["best_score"] = best_score
    summary_path = os.path.join(
        Config.PRED_ABLATION_OUTPUT_DIR,
        f"ablation_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json",
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved ablation summary to {summary_path}")


# DO NOT change the mapping of class IDs to names
class_dict = {
    0: "Arctic fox, white fox, Alopex lagopus",
    1: "Australian terrier",
    2: "altar",
    3: "ballplayer, baseball player",
    4: "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    5: "bearskin, busby, shako",
    6: "bell pepper",
    7: "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    8: "bustard",
    9: "clog, geta, patten, sabot",
    10: "cocktail shaker",
    11: "confectionery, confectionary, candy store",
    12: "coral fungus",
    13: "corn",
    14: "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    15: "dam, dike, dyke",
    16: "desktop computer",
    17: "four-poster",
    18: "gas pump, gasoline pump, petrol pump, island dispenser",
    19: "goldfish, Carassius auratus",
    20: "hair spray",
    21: "harvester, reaper",
    22: "isopod",
    23: "lotion",
    24: "mashed potato",
    25: "meerkat, mierkat",
    26: "mud turtle",
    27: "necklace",
    28: "oboe, hautboy, hautbois",
    29: "orangutan, orang, orangutang, Pongo pygmaeus",
    30: "otter",
    31: "pencil sharpener",
    32: "plane, carpenter's plane, woodworking plane",
    33: "pop bottle, soda bottle",
    34: "puffer, pufferfish, blowfish, globefish",
    35: "quilt, comforter, comfort, puff",
    36: "screen, CRT screen",
    37: "silky terrier, Sydney silky",
    38: "sleeping bag",
    39: "snow leopard, ounce, Panthera uncia",
    40: "spatula",
    41: "spotted salamander, Ambystoma maculatum",
    42: "strawberry",
    43: "tarantula",
    44: "thresher, thrasher, threshing machine",
    45: "unicycle, monocycle",
    46: "warplane, military plane",
    47: "whiptail, whiptail lizard",
    48: "wood rabbit, cottontail, cottontail rabbit",
    49: "yurt",
}

parser = argparse.ArgumentParser(description="Predict on testset with CLIP.")
parser.add_argument(
    "--run-ablations",
    action="store_true",
    help="Run Exp A->E ablation ladder and save each prediction file.",
)
args = parser.parse_args()

transform = get_eval_transform()
model = CLIP(
    encoder_type=Config.IMAGE_ENCODER,
    embed_dim=Config.EMBED_DIM,
    temperature=Config.TEMPERATURE,
    pretrained=False,
    trainable_image_blocks=Config.UNFREEZE_IMAGE_BLOCKS,
    trainable_text_layers=Config.UNFREEZE_TEXT_LAYERS,
)
state_dict = load_file(Config.BEST_MODEL_PATH)
model.load_state_dict(state_dict)
model = model.to("cuda")
model.eval()

# Dataset and DataLoader for flat testset
test_set = TestDataset(Config.TEST_DIR, transform=transform)
dataloader = DataLoader(
    test_set,
    batch_size=Config.EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
)

class_names = [class_dict[i] for i in range(len(class_dict))]

if args.run_ablations or Config.PRED_RUN_ABLATIONS:
    run_ablation_sweep(model=model, dataloader=dataloader, class_names=class_names)

default_tta_modes = Config.PRED_TTA_MODES if Config.PRED_ENABLE_TTA else ["base"]
results, diagnostics = predict_with_recipe(
    model=model,
    dataloader=dataloader,
    class_names=class_names,
    templates=Config.EVAL_TEXT_TEMPLATES,
    template_weights=Config.EVAL_TEMPLATE_WEIGHTS,
    class_prompt_variants=Config.TESTSET_CLASS_PROMPT_VARIANTS,
    tta_modes=default_tta_modes,
)
save_predictions(Config.PRED_PATH, results)
print(
    f"Saved {len(results)} predictions to {Config.PRED_PATH} "
    f"(avg_top1_conf={diagnostics['avg_top1_confidence']:.4f}, "
    f"avg_margin={diagnostics['avg_top1_top2_margin']:.4f})"
)
