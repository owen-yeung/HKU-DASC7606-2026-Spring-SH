import argparse
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

# CELL 3
## Zero-shot on ImageNet Validation Set
imgnet = load_imagenet(
    Config.IMGNET_DIR,
    train_transform=transform,
    val_transform=transform,
)
val_set = imgnet["val"]
class_names = imgnet["full"].classes
print(
    f"Loaded ImageNet validation set with {len(val_set)} samples and {len(class_names)} classes."
)

imagenet_result = topk_evaluate(
    model,
    val_set,
    class_names,
    batch_size=Config.EVAL_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    text_templates=Config.EVAL_TEXT_TEMPLATES,
    return_details=True,
)
save_eval_result("imagenet_val", imagenet_result, val_set)

# CELL 4
## Zero-shot on CIFAR-10
cifar = load_hf_dataset("uoft-cs/cifar10", transform=transform)
cifar = concatenate_datasets([cifar["train"], cifar["test"]])
cifar = cifar.rename_column("img", "image")
class_names = cifar.features["label"].names
print(
    f"Loaded CIFAR-10 set with {len(cifar)} samples and {len(class_names)} classes."
)

cifar10_result = topk_evaluate(
    model,
    cifar,
    class_names,
    batch_size=Config.EVAL_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    topk=[1, 2, 3],
    text_templates=Config.EVAL_TEXT_TEMPLATES,
    return_details=True,
)
save_eval_result("cifar10_all", cifar10_result, cifar)

# CELL 5
## Zero-shot on CIFAR-100
cifar = load_hf_dataset("uoft-cs/cifar100", transform=transform)
cifar = concatenate_datasets([cifar["train"], cifar["test"]])
cifar = cifar.rename_column("img", "image")
cifar = cifar.rename_column("fine_label", "label")
class_names = cifar.features["label"].names
print(
    f"Loaded CIFAR-100 set with {len(cifar)} samples and {len(class_names)} classes."
)

cifar100_result = topk_evaluate(
    model,
    cifar,
    class_names,
    batch_size=Config.EVAL_BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    text_templates=Config.EVAL_TEXT_TEMPLATES,
    return_details=True,
)
save_eval_result("cifar100_all", cifar100_result, cifar)