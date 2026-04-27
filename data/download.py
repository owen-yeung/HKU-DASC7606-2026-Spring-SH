"""
Download 500 images per class × 500 classes from ImageNet-1k (streaming).
1. HuggingFace: <https://huggingface.co/datasets/ILSVRC/imagenet-1k>
2. Homepage: <https://www.image-net.org/>

Usage:
    hf auth login   # required — imagenet-1k is gated
    python download.py
    python download.py --start-image 230000
"""

import os
import random
import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SEED = 42
NUM_CLASSES = 500
IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * IMAGES_PER_CLASS
# Default resume point for interrupted downloads.
DEFAULT_RESUME_IMAGES = 23_308
OUTPUT_DIR = Path(__file__).parent.joinpath("imagenet")
OUTPUT_DIR.mkdir(exist_ok=True)
random.seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a deterministic subset of ImageNet images."
    )
    parser.add_argument(
        "--start-image",
        type=int,
        default=None,
        help=(
            "Start downloading from global selected-image index X (0-based) "
            "in the deterministic shuffled stream."
        ),
    )
    return parser.parse_args()


args = parse_args()


print("Loading dataset in streaming mode …")
# https://huggingface.co/docs/datasets/main/en/stream
ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
class_label = ds.features["label"]

# Randomly select target classes to collect (e.g. 500 random classes out of 1000 total)
target_label_ids = set(random.sample(range(class_label.num_classes), NUM_CLASSES))

# Shuffle the stream with a fixed seed so sampling is reproducible.
ds = ds.shuffle(seed=SEED, buffer_size=10_000)

# label_id -> saved count
saved_count = {}
for label_id in range(class_label.num_classes):
    class_name = class_label.int2str(label_id)
    class_dir = OUTPUT_DIR.joinpath(class_name)
    if class_dir.exists():
        saved_count[label_id] = len(list(class_dir.glob("*.jpg")))

saved_from_disk = sum(saved_count.values())
if args.start_image is not None:
    if args.start_image < 0 or args.start_image >= NUM_IMAGES:
        raise ValueError(f"--start-image must be in [0, {NUM_IMAGES - 1}]")
    start_image_idx = args.start_image
    total_saved = 0
    print(f"Skipping to global selected-image index {start_image_idx}.")
elif saved_from_disk >= DEFAULT_RESUME_IMAGES:
    start_image_idx = 0
    total_saved = saved_from_disk
else:
    start_image_idx = 0
    total_saved = DEFAULT_RESUME_IMAGES
    print(
        f"Resuming from default image index {DEFAULT_RESUME_IMAGES}, "
        f"but only found {saved_from_disk} images on disk."
    )
print(f"Starting from {total_saved}/{NUM_IMAGES} images saved.")

with tqdm(total=NUM_IMAGES, desc="Saving images", unit="img", initial=total_saved) as pbar:
    selected_seen = 0
    stream_class_count = {}
    for item in ds:
        label_id = item["label"]

        # Skip non-target classes
        if label_id not in target_label_ids:
            continue

        # Deterministic selected-image counter independent of local files.
        stream_cnt = stream_class_count.get(label_id, 0)
        if stream_cnt >= IMAGES_PER_CLASS:
            continue
        stream_class_count[label_id] = stream_cnt + 1

        if selected_seen < start_image_idx:
            selected_seen += 1
            continue

        # Skip classes that are already have enough images
        cnt = saved_count.get(label_id, 0)
        if cnt >= IMAGES_PER_CLASS:
            selected_seen += 1
            continue

        # Save the image
        class_name = class_label.int2str(label_id)
        class_dir = OUTPUT_DIR.joinpath(class_name)
        class_dir.mkdir(exist_ok=True)

        img_path = class_dir.joinpath(f"{cnt:04d}.jpg")
        img = item["image"]
        img.save(img_path)

        saved_count[label_id] = cnt + 1
        total_saved += 1
        selected_seen += 1
        pbar.update(1)

        if total_saved >= NUM_IMAGES:
            break


print(f"Saved {total_saved} images for {len(saved_count)} classes to {OUTPUT_DIR}")
# Close to avoid stream hanging
os._exit(0)
