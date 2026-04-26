import json
import os
from datetime import datetime

from safetensors.torch import load_file
from datasets import concatenate_datasets
from config import Config
from data.dataset import get_transform, load_imagenet, load_hf_dataset
from model.clip import CLIP
from utils import topk_evaluate

transform = get_transform()
model = CLIP(
    encoder_type=Config.IMAGE_ENCODER,
    embed_dim=Config.EMBED_DIM,
    temperature=Config.TEMPERATURE,
    pretrained=False,
)
state_dict = load_file(Config.BEST_MODEL_PATH)
model.load_state_dict(state_dict)
model = model.to("cuda")

eval_output_dir = os.path.join(
    Config.OUTPUT_DIR, "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(eval_output_dir, exist_ok=True)


def save_eval_result(dataset_name, result):
    output_path = os.path.join(eval_output_dir, f"{dataset_name}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(
        f"[{dataset_name}] metrics={result['metrics']} misclassified={result['num_misclassified']} output={output_path}"
    )

# CELL 3
## Zero-shot on ImageNet Validation Set
imgnet = load_imagenet(Config.IMGNET_DIR, transform=transform)
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
    text_template=Config.EVAL_TEXT_TEMPLATE,
    return_details=True,
)
save_eval_result("imagenet_val", imagenet_result)

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
    text_template=Config.EVAL_TEXT_TEMPLATE,
    return_details=True,
)
save_eval_result("cifar10_all", cifar10_result)

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
    text_template=Config.EVAL_TEXT_TEMPLATE,
    return_details=True,
)
save_eval_result("cifar100_all", cifar100_result)