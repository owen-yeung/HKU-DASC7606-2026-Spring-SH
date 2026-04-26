"""
Download 500 images per class × 500 classes from ImageNet-1k (streaming).
1. HuggingFace: <https://huggingface.co/datasets/ILSVRC/imagenet-1k>
2. Homepage: <https://www.image-net.org/>

Usage:
    hf auth login   # required — imagenet-1k is gated
    python download.py
"""

import os
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SEED = 42
NUM_CLASSES = 500
IMAGES_PER_CLASS = 500
NUM_IMAGES = NUM_CLASSES * IMAGES_PER_CLASS
OUTPUT_DIR = Path(__file__).parent.joinpath("imagenet")
OUTPUT_DIR.mkdir(exist_ok=True)
random.seed(SEED)


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
total_saved = 0

with tqdm(total=NUM_IMAGES, desc="Saving images", unit="img") as pbar:
    for item in ds:
        label_id = item["label"]

        # Skip non-target classes
        if label_id not in target_label_ids:
            continue

        # Skip classes that are already have enough images
        cnt = saved_count.get(label_id, 0)
        if cnt >= IMAGES_PER_CLASS:
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
        pbar.update(1)

        if total_saved >= NUM_IMAGES:
            break


print(f"Saved {total_saved} images for {len(saved_count)} classes to {OUTPUT_DIR}")
# Close to avoid stream hanging
os._exit(0)
