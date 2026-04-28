import argparse
import csv
import json
import os
from datetime import datetime

import torch
from PIL import Image
from safetensors.torch import load_file
from datasets import concatenate_datasets
from config import Config
from data.dataset import get_eval_transform, load_hf_dataset, load_imagenet
from model.clip import CLIP
from utils import topk_evaluate
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

parser = argparse.ArgumentParser(description="Evaluate CLIP on multiple datasets.")
parser.add_argument(
    "--weights",
    choices=["pretrained", "finetuned"],
    default="finetuned",
    help="Choose whether to use pretrained backbone/text weights or a fine-tuned checkpoint.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=Config.BEST_MODEL_PATH,
    help="Path to fine-tuned .safetensors checkpoint. Used only when --weights finetuned.",
)
parser.add_argument(
    "--run-ablations",
    action="store_true",
    help="Run Exp A->E ablation sweep on eval datasets and select by weighted top-1 score.",
)
parser.add_argument(
    "--run-checkpoint-series",
    action="store_true",
    help="Evaluate all checkpoint-* directories using best ablation recipe.",
)
parser.add_argument(
    "--ablation-summary",
    type=str,
    default=None,
    help="Path to eval_ablation_summary.json. If omitted, uses latest in eval pushable dir.",
)
parser.add_argument(
    "--checkpoint-root",
    type=str,
    default=None,
    help="Directory containing checkpoint-* folders. If omitted, inferred from BEST_MODEL_PATH.",
)
args = parser.parse_args()

transform = get_eval_transform()
use_finetuned = args.weights == "finetuned"
model = CLIP(
    encoder_type=Config.IMAGE_ENCODER,
    embed_dim=Config.EMBED_DIM,
    temperature=Config.TEMPERATURE,
    pretrained=not use_finetuned,
    trainable_image_blocks=Config.UNFREEZE_IMAGE_BLOCKS if use_finetuned else 0,
    trainable_text_layers=Config.UNFREEZE_TEXT_LAYERS if use_finetuned else 0,
)
if use_finetuned:
    state_dict = load_file(args.checkpoint)
    model.load_state_dict(state_dict)
model = model.to("cuda")

eval_output_dir = os.path.join(
    Config.OUTPUT_DIR, "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(eval_output_dir, exist_ok=True)
pushable_output_dir = os.path.join(
    Config.EVAL_PUSHABLE_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(pushable_output_dir, exist_ok=True)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
PREVIEW_TOP_N = 10


def _to_python_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _extract_tensor_and_label(sample):
    if isinstance(sample, dict):
        image = sample["image"]
        label = sample["label"]
    else:
        image, label = sample
    return image, int(_to_python_value(label))


def _save_tensor_image(tensor_image, save_path):
    image = tensor_image.detach().cpu()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).mul(255).byte().numpy()
    Image.fromarray(image).save(save_path)


def _write_review_html(dataset_name, small_result, images_dir, html_path):
    rows = []
    for sample in small_result["misclassified_samples_top_n"]:
        preview_path = sample.get("image_preview_path")
        img_cell = (
            f'<img src="{preview_path}" alt="sample {sample["sample_index"]}" loading="lazy" />'
            if preview_path
            else "<span class='na'>N/A</span>"
        )
        rows.append(
            "<tr>"
            f"<td>{sample['sample_index']}</td>"
            f"<td>{img_cell}</td>"
            f"<td>{sample['true_label_name']}</td>"
            f"<td>{sample['predicted_label_name']}</td>"
            f"<td>{sample['predicted_confidence']:.4f}</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{dataset_name} Misclassified Samples</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; }}
    h1 {{ margin-bottom: 0; }}
    p {{ margin-top: 8px; color: #444; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: middle; }}
    th {{ text-align: left; background: #f6f6f6; }}
    img {{ width: 96px; height: 96px; object-fit: contain; background: #fafafa; }}
    .na {{ color: #999; }}
  </style>
</head>
<body>
  <h1>{dataset_name} - Misclassified Samples</h1>
  <p>
    Top-{small_result["top_n"]} misclassified samples.
    Open folder <code>{images_dir}</code> for raw preview PNGs.
  </p>
  <table>
    <thead>
      <tr>
        <th>Sample Index</th>
        <th>Image</th>
        <th>True Label</th>
        <th>Predicted Label</th>
        <th>Confidence</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)


def save_eval_result(dataset_name, result, dataset):
    output_path = os.path.join(eval_output_dir, f"{dataset_name}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    top_n = int(Config.EVAL_PUSHABLE_TOP_N)
    small_result = {
        "dataset": dataset_name,
        "metrics": result["metrics"],
        "num_samples": result["num_samples"],
        "num_misclassified": result["num_misclassified"],
        "top_n": top_n,
        "image_preview_top_n": PREVIEW_TOP_N,
        "misclassified_samples_top_n": result["misclassified_samples"][:top_n],
    }
    images_dir = f"{dataset_name}.sample_images"
    images_abs_dir = os.path.join(pushable_output_dir, images_dir)
    os.makedirs(images_abs_dir, exist_ok=True)

    for sample in small_result["misclassified_samples_top_n"][:PREVIEW_TOP_N]:
        sample_index = int(sample["sample_index"])
        sample_data = dataset[sample_index]
        image_tensor, _ = _extract_tensor_and_label(sample_data)
        image_filename = f"{sample_index:06d}.png"
        image_abs_path = os.path.join(images_abs_dir, image_filename)
        _save_tensor_image(image_tensor, image_abs_path)
        sample["image_preview_path"] = f"{images_dir}/{image_filename}"

    pushable_path = os.path.join(pushable_output_dir, f"{dataset_name}.small.json")
    with open(pushable_path, "w") as f:
        json.dump(small_result, f, indent=2)

    review_html_path = os.path.join(pushable_output_dir, f"{dataset_name}.review.html")
    _write_review_html(
        dataset_name=dataset_name,
        small_result=small_result,
        images_dir=images_dir,
        html_path=review_html_path,
    )

    print(
        f"[{dataset_name}] metrics={result['metrics']} misclassified={result['num_misclassified']} full_output={output_path} pushable_output={pushable_path} review_html={review_html_path}"
    )
    return {
        "full_output": output_path,
        "small_output": pushable_path,
        "review_html": review_html_path,
    }

def _evaluate_all_datasets(recipe_name, templates, template_weights, class_prompt_variants, tta_modes):
    dataset_suffix = "" if recipe_name == "default" else f".{recipe_name}"

    imgnet = load_imagenet(
        Config.IMGNET_DIR,
        train_transform=transform,
        val_transform=transform,
    )
    val_set = imgnet["val"]
    class_names = imgnet["full"].classes
    print(
        f"[{recipe_name}] Loaded ImageNet validation set with {len(val_set)} samples and {len(class_names)} classes."
    )
    imagenet_result = topk_evaluate(
        model,
        val_set,
        class_names,
        batch_size=Config.EVAL_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        text_templates=templates,
        template_weights=template_weights,
        class_prompt_variants=class_prompt_variants,
        tta_modes=tta_modes,
        return_details=True,
    )
    imagenet_paths = save_eval_result(
        f"imagenet_val{dataset_suffix}", imagenet_result, val_set
    )

    cifar10 = load_hf_dataset("uoft-cs/cifar10", transform=transform)
    cifar10 = concatenate_datasets([cifar10["train"], cifar10["test"]])
    cifar10 = cifar10.rename_column("img", "image")
    class_names = cifar10.features["label"].names
    print(
        f"[{recipe_name}] Loaded CIFAR-10 set with {len(cifar10)} samples and {len(class_names)} classes."
    )
    cifar10_result = topk_evaluate(
        model,
        cifar10,
        class_names,
        batch_size=Config.EVAL_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        topk=[1, 2, 3],
        text_templates=templates,
        template_weights=template_weights,
        class_prompt_variants=class_prompt_variants,
        tta_modes=tta_modes,
        return_details=True,
    )
    cifar10_paths = save_eval_result(f"cifar10_all{dataset_suffix}", cifar10_result, cifar10)

    cifar100 = load_hf_dataset("uoft-cs/cifar100", transform=transform)
    cifar100 = concatenate_datasets([cifar100["train"], cifar100["test"]])
    cifar100 = cifar100.rename_column("img", "image")
    cifar100 = cifar100.rename_column("fine_label", "label")
    class_names = cifar100.features["label"].names
    print(
        f"[{recipe_name}] Loaded CIFAR-100 set with {len(cifar100)} samples and {len(class_names)} classes."
    )
    cifar100_result = topk_evaluate(
        model,
        cifar100,
        class_names,
        batch_size=Config.EVAL_BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        text_templates=templates,
        template_weights=template_weights,
        class_prompt_variants=class_prompt_variants,
        tta_modes=tta_modes,
        return_details=True,
    )
    cifar100_paths = save_eval_result(
        f"cifar100_all{dataset_suffix}", cifar100_result, cifar100
    )

    return {
        "imagenet_top1": float(imagenet_result["metrics"]["top1_accuracy"]),
        "cifar10_top1": float(cifar10_result["metrics"]["top1_accuracy"]),
        "cifar100_top1": float(cifar100_result["metrics"]["top1_accuracy"]),
        "artifact_paths": {
            "imagenet_val": imagenet_paths,
            "cifar10_all": cifar10_paths,
            "cifar100_all": cifar100_paths,
        },
    }


def _compute_weighted_score(metric_triplet):
    w = Config.EVAL_ABLATION_SCORE_WEIGHTS
    return (
        w["cifar10_top1"] * metric_triplet["cifar10_top1"]
        + w["cifar100_top1"] * metric_triplet["cifar100_top1"]
        + w["imagenet_top1"] * metric_triplet["imagenet_top1"]
    )


def run_eval_ablation_sweep():
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
            "variants": Config.EVAL_CLASS_PROMPT_VARIANTS,
            "tta_modes": ["base"],
        },
        {
            "name": "exp_d_plus_tta",
            "templates": Config.EVAL_TEXT_TEMPLATES,
            "weights": [],
            "variants": Config.EVAL_CLASS_PROMPT_VARIANTS,
            "tta_modes": Config.EVAL_TTA_MODES,
        },
        {
            "name": "exp_e_plus_weights",
            "templates": Config.EVAL_TEXT_TEMPLATES,
            "weights": Config.EVAL_TEMPLATE_WEIGHTS,
            "variants": Config.EVAL_CLASS_PROMPT_VARIANTS,
            "tta_modes": Config.EVAL_TTA_MODES,
        },
    ]

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "score_weights": Config.EVAL_ABLATION_SCORE_WEIGHTS,
        "experiments": [],
    }
    best_name = None
    best_score = None
    for exp in exps:
        print(f"Running eval ablation: {exp['name']}")
        metrics = _evaluate_all_datasets(
            recipe_name=exp["name"],
            templates=exp["templates"],
            template_weights=exp["weights"],
            class_prompt_variants=exp["variants"],
            tta_modes=exp["tta_modes"],
        )
        score = _compute_weighted_score(metrics)
        summary["experiments"].append(
            {
                "name": exp["name"],
                "metrics": metrics,
                "score": score,
                "templates": exp["templates"],
                "template_weights": exp["weights"],
                "tta_modes": exp["tta_modes"],
                "artifact_paths": metrics["artifact_paths"],
            }
        )
        if best_score is None or score > best_score:
            best_score = score
            best_name = exp["name"]

    summary["best_experiment"] = best_name
    summary["best_score"] = best_score
    summary_path = os.path.join(pushable_output_dir, "eval_ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved eval ablation summary to {summary_path}")
    _write_eval_ablation_report_bundle(summary, summary_path)


def _make_rel(path):
    return os.path.relpath(path, pushable_output_dir)


def _write_eval_ablation_report_bundle(summary, summary_path):
    rows = sorted(summary["experiments"], key=lambda x: x["score"], reverse=True)
    csv_path = os.path.join(pushable_output_dir, "eval_ablation_leaderboard.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "experiment",
                "score",
                "cifar10_top1",
                "cifar100_top1",
                "imagenet_top1",
                "imagenet_small_json",
                "cifar10_small_json",
                "cifar100_small_json",
            ]
        )
        for i, row in enumerate(rows, start=1):
            artifacts = row.get("artifact_paths", {})
            writer.writerow(
                [
                    i,
                    row["name"],
                    f"{row['score']:.6f}",
                    f"{row['metrics']['cifar10_top1']:.6f}",
                    f"{row['metrics']['cifar100_top1']:.6f}",
                    f"{row['metrics']['imagenet_top1']:.6f}",
                    _make_rel(artifacts["imagenet_val"]["small_output"]),
                    _make_rel(artifacts["cifar10_all"]["small_output"]),
                    _make_rel(artifacts["cifar100_all"]["small_output"]),
                ]
            )
    print(f"Saved eval ablation leaderboard CSV to {csv_path}")

    plot_path = os.path.join(pushable_output_dir, "eval_ablation_scores.png")
    if plt is not None:
        names = [row["name"] for row in rows]
        scores = [row["score"] for row in rows]
        plt.figure(figsize=(10, 5))
        plt.bar(names, scores)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Weighted score")
        plt.title("Eval Ablation Leaderboard")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved eval ablation score plot to {plot_path}")
    else:
        plot_path = None

    md_path = os.path.join(pushable_output_dir, "eval_ablation_report.md")
    with open(md_path, "w") as f:
        f.write("# Eval Ablation Report\n\n")
        f.write(f"- Timestamp: `{summary['timestamp']}`\n")
        f.write(f"- Best experiment: `{summary['best_experiment']}`\n")
        f.write(f"- Best weighted score: `{summary['best_score']:.6f}`\n")
        f.write(f"- Score weights: `{summary['score_weights']}`\n")
        f.write(f"- JSON summary: `{_make_rel(summary_path)}`\n")
        f.write(f"- CSV leaderboard: `{_make_rel(csv_path)}`\n")
        if plot_path:
            f.write(f"- Score plot: `{_make_rel(plot_path)}`\n")
        f.write("\n## Leaderboard\n\n")
        f.write("| Rank | Experiment | Score | CIFAR10 Top1 | CIFAR100 Top1 | ImageNet Top1 |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for i, row in enumerate(rows, start=1):
            f.write(
                f"| {i} | `{row['name']}` | {row['score']:.6f} | "
                f"{row['metrics']['cifar10_top1']:.6f} | "
                f"{row['metrics']['cifar100_top1']:.6f} | "
                f"{row['metrics']['imagenet_top1']:.6f} |\n"
            )
        f.write("\n## Per-experiment Artifacts\n\n")
        for row in rows:
            a = row["artifact_paths"]
            f.write(f"### `{row['name']}`\n")
            f.write(f"- ImageNet small JSON: `{_make_rel(a['imagenet_val']['small_output'])}`\n")
            f.write(f"- CIFAR10 small JSON: `{_make_rel(a['cifar10_all']['small_output'])}`\n")
            f.write(f"- CIFAR100 small JSON: `{_make_rel(a['cifar100_all']['small_output'])}`\n")
            f.write(f"- ImageNet review HTML: `{_make_rel(a['imagenet_val']['review_html'])}`\n")
            f.write(f"- CIFAR10 review HTML: `{_make_rel(a['cifar10_all']['review_html'])}`\n")
            f.write(f"- CIFAR100 review HTML: `{_make_rel(a['cifar100_all']['review_html'])}`\n\n")
    print(f"Saved eval ablation markdown report to {md_path}")

    html_path = os.path.join(pushable_output_dir, "eval_ablation_report.html")
    rows_html = []
    for i, row in enumerate(rows, start=1):
        rows_html.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><code>{row['name']}</code></td>"
            f"<td>{row['score']:.6f}</td>"
            f"<td>{row['metrics']['cifar10_top1']:.6f}</td>"
            f"<td>{row['metrics']['cifar100_top1']:.6f}</td>"
            f"<td>{row['metrics']['imagenet_top1']:.6f}</td>"
            "</tr>"
        )
    plot_img = (
        f'<p><img src="{os.path.basename(plot_path)}" alt="ablation scores" loading="lazy"/></p>'
        if plot_path
        else "<p><em>matplotlib unavailable: score plot not generated.</em></p>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Eval Ablation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ text-align: left; background: #f6f6f6; }}
    code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 4px; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Eval Ablation Report</h1>
  <p>Best experiment: <code>{summary['best_experiment']}</code> (score={summary['best_score']:.6f})</p>
  <p>Score weights: <code>{summary['score_weights']}</code></p>
  {plot_img}
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>Experiment</th><th>Score</th><th>CIFAR10 Top1</th><th>CIFAR100 Top1</th><th>ImageNet Top1</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved eval ablation HTML report to {html_path}")


def _infer_checkpoint_root(best_model_path):
    norm = os.path.normpath(best_model_path)
    parent = os.path.dirname(norm)
    parent_name = os.path.basename(parent)
    if parent_name.startswith("checkpoint-"):
        return os.path.dirname(parent)
    return parent


def _extract_checkpoint_step(checkpoint_dir_name):
    try:
        return int(checkpoint_dir_name.split("checkpoint-")[-1])
    except Exception:
        return -1


def _find_latest_ablation_summary(base_dir):
    candidates = []
    for root, _, files in os.walk(base_dir):
        if "eval_ablation_summary.json" in files:
            abs_path = os.path.join(root, "eval_ablation_summary.json")
            candidates.append((os.path.getmtime(abs_path), abs_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_best_ablation_recipe(summary_path):
    with open(summary_path, "r") as f:
        summary = json.load(f)
    best_name = summary.get("best_experiment")
    if not best_name:
        raise ValueError(f"Missing best_experiment in ablation summary: {summary_path}")
    experiments = summary.get("experiments", [])
    best_exp = None
    for exp in experiments:
        if exp.get("name") == best_name:
            best_exp = exp
            break
    if best_exp is None:
        raise ValueError(
            f"Could not find experiment '{best_name}' in ablation summary: {summary_path}"
        )
    return {
        "name": best_name,
        "templates": best_exp.get("templates", Config.EVAL_TEXT_TEMPLATES),
        "template_weights": best_exp.get("template_weights", []),
        "tta_modes": best_exp.get("tta_modes", ["base"]),
        "class_prompt_variants": best_exp.get(
            "class_prompt_variants", Config.EVAL_CLASS_PROMPT_VARIANTS
        ),
    }


def _write_checkpoint_series_report_bundle(summary, summary_path):
    rows = sorted(summary["checkpoints"], key=lambda x: x["step"])
    csv_path = os.path.join(pushable_output_dir, "checkpoint_series_leaderboard.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "checkpoint",
                "score",
                "cifar10_top1",
                "cifar100_top1",
                "imagenet_top1",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["step"],
                    row["checkpoint_name"],
                    f"{row['score']:.6f}",
                    f"{row['metrics']['cifar10_top1']:.6f}",
                    f"{row['metrics']['cifar100_top1']:.6f}",
                    f"{row['metrics']['imagenet_top1']:.6f}",
                ]
            )
    print(f"Saved checkpoint series CSV to {csv_path}")

    plot_path = os.path.join(pushable_output_dir, "checkpoint_series_scores.png")
    if plt is not None and rows:
        steps = [r["step"] for r in rows]
        weighted = [r["score"] for r in rows]
        cifar10 = [r["metrics"]["cifar10_top1"] for r in rows]
        cifar100 = [r["metrics"]["cifar100_top1"] for r in rows]
        imagenet = [r["metrics"]["imagenet_top1"] for r in rows]
        plt.figure(figsize=(10, 5))
        plt.plot(steps, weighted, marker="o", label="weighted_score")
        plt.plot(steps, cifar10, marker="o", label="cifar10_top1")
        plt.plot(steps, cifar100, marker="o", label="cifar100_top1")
        plt.plot(steps, imagenet, marker="o", label="imagenet_top1")
        plt.xlabel("Checkpoint step")
        plt.ylabel("Score")
        plt.title("Checkpoint Series Evaluation")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved checkpoint series plot to {plot_path}")
    else:
        plot_path = None

    md_path = os.path.join(pushable_output_dir, "checkpoint_series_report.md")
    with open(md_path, "w") as f:
        f.write("# Checkpoint Series Evaluation Report\n\n")
        f.write(f"- Timestamp: `{summary['timestamp']}`\n")
        f.write(f"- Checkpoint root: `{summary['checkpoint_root']}`\n")
        f.write(f"- Ablation recipe: `{summary['ablation_recipe']['name']}`\n")
        f.write(f"- Ablation summary source: `{summary['ablation_summary_path']}`\n")
        f.write(f"- JSON summary: `{os.path.relpath(summary_path, pushable_output_dir)}`\n")
        f.write(f"- CSV leaderboard: `{os.path.relpath(csv_path, pushable_output_dir)}`\n")
        if plot_path:
            f.write(f"- Trend plot: `{os.path.relpath(plot_path, pushable_output_dir)}`\n")
        f.write("\n## Metrics by Checkpoint\n\n")
        f.write("| Step | Checkpoint | Weighted Score | CIFAR10 Top1 | CIFAR100 Top1 | ImageNet Top1 |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['step']} | `{row['checkpoint_name']}` | {row['score']:.6f} | "
                f"{row['metrics']['cifar10_top1']:.6f} | {row['metrics']['cifar100_top1']:.6f} | "
                f"{row['metrics']['imagenet_top1']:.6f} |\n"
            )
    print(f"Saved checkpoint series markdown report to {md_path}")

    html_path = os.path.join(pushable_output_dir, "checkpoint_series_report.html")
    rows_html = []
    for row in rows:
        rows_html.append(
            "<tr>"
            f"<td>{row['step']}</td>"
            f"<td><code>{row['checkpoint_name']}</code></td>"
            f"<td>{row['score']:.6f}</td>"
            f"<td>{row['metrics']['cifar10_top1']:.6f}</td>"
            f"<td>{row['metrics']['cifar100_top1']:.6f}</td>"
            f"<td>{row['metrics']['imagenet_top1']:.6f}</td>"
            "</tr>"
        )
    plot_img = (
        f'<p><img src="{os.path.basename(plot_path)}" alt="checkpoint trend plot" loading="lazy"/></p>'
        if plot_path
        else "<p><em>matplotlib unavailable: trend plot not generated.</em></p>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Checkpoint Series Evaluation</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; }}
    th {{ text-align: left; background: #f6f6f6; }}
    code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 4px; }}
    img {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Checkpoint Series Evaluation</h1>
  <p>Ablation recipe: <code>{summary['ablation_recipe']['name']}</code></p>
  <p>Score weights: <code>{summary['score_weights']}</code></p>
  {plot_img}
  <table>
    <thead>
      <tr>
        <th>Step</th><th>Checkpoint</th><th>Weighted Score</th><th>CIFAR10 Top1</th><th>CIFAR100 Top1</th><th>ImageNet Top1</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved checkpoint series HTML report to {html_path}")


def run_checkpoint_series_eval():
    checkpoint_root = args.checkpoint_root or _infer_checkpoint_root(args.checkpoint)
    if not os.path.isdir(checkpoint_root):
        raise FileNotFoundError(f"Checkpoint root does not exist: {checkpoint_root}")

    ablation_summary_path = args.ablation_summary
    if not ablation_summary_path:
        ablation_summary_path = _find_latest_ablation_summary(Config.EVAL_PUSHABLE_DIR)
    if not ablation_summary_path:
        raise FileNotFoundError(
            "Could not find eval_ablation_summary.json. Provide --ablation-summary explicitly."
        )

    recipe = _load_best_ablation_recipe(ablation_summary_path)
    print(
        f"Using best ablation recipe '{recipe['name']}' from {ablation_summary_path}"
    )

    checkpoint_dirs = []
    for entry in os.listdir(checkpoint_root):
        if not entry.startswith("checkpoint-"):
            continue
        model_path = os.path.join(checkpoint_root, entry, "model.safetensors")
        if os.path.isfile(model_path):
            checkpoint_dirs.append((entry, model_path))
    checkpoint_dirs.sort(key=lambda x: _extract_checkpoint_step(x[0]))
    if not checkpoint_dirs:
        raise RuntimeError(f"No checkpoint-*/model.safetensors found under {checkpoint_root}")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "checkpoint_root": checkpoint_root,
        "ablation_summary_path": ablation_summary_path,
        "ablation_recipe": recipe,
        "score_weights": Config.EVAL_ABLATION_SCORE_WEIGHTS,
        "checkpoints": [],
    }
    best_ckpt = None
    best_score = None
    for checkpoint_name, checkpoint_path in checkpoint_dirs:
        print(f"Evaluating checkpoint series item: {checkpoint_name}")
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict)
        metrics = _evaluate_all_datasets(
            recipe_name=f"series_{checkpoint_name}",
            templates=recipe["templates"],
            template_weights=recipe["template_weights"],
            class_prompt_variants=recipe["class_prompt_variants"],
            tta_modes=recipe["tta_modes"],
        )
        score = _compute_weighted_score(metrics)
        record = {
            "checkpoint_name": checkpoint_name,
            "checkpoint_path": checkpoint_path,
            "step": _extract_checkpoint_step(checkpoint_name),
            "metrics": metrics,
            "score": score,
            "artifact_paths": metrics["artifact_paths"],
        }
        summary["checkpoints"].append(record)
        if best_score is None or score > best_score:
            best_score = score
            best_ckpt = checkpoint_name

    summary["best_checkpoint"] = best_ckpt
    summary["best_score"] = best_score
    summary_path = os.path.join(pushable_output_dir, "checkpoint_series_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved checkpoint series summary to {summary_path}")
    _write_checkpoint_series_report_bundle(summary, summary_path)


if args.run_ablations or Config.EVAL_RUN_ABLATIONS:
    run_eval_ablation_sweep()
if args.run_checkpoint_series:
    run_checkpoint_series_eval()
elif not (args.run_ablations or Config.EVAL_RUN_ABLATIONS):
    default_tta_modes = Config.EVAL_TTA_MODES if Config.EVAL_ENABLE_TTA else ["base"]
    _evaluate_all_datasets(
        recipe_name="default",
        templates=Config.EVAL_TEXT_TEMPLATES,
        template_weights=Config.EVAL_TEMPLATE_WEIGHTS,
        class_prompt_variants=Config.EVAL_CLASS_PROMPT_VARIANTS,
        tta_modes=default_tta_modes,
    )