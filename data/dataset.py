"""
Handles loading and preprocessing datasets
"""

import torch
import torchvision.transforms as T
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from datasets import load_dataset
from functools import partial
from PIL import UnidentifiedImageError
import warnings


class SafeImageFolder(ImageFolder):
    """
    ImageFolder variant that skips corrupted/unreadable image files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._warned_paths = set()

    def __getitem__(self, index):
        total = len(self.samples)
        for offset in range(total):
            current_index = (index + offset) % total
            path, target = self.samples[current_index]
            try:
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                if self.target_transform is not None:
                    target = self.target_transform(target)
                return sample, target
            except (UnidentifiedImageError, OSError, ValueError) as err:
                if path not in self._warned_paths:
                    self._warned_paths.add(path)
                    warnings.warn(
                        f"Skipping unreadable image file: {path} ({type(err).__name__})",
                        RuntimeWarning,
                    )
                continue

        raise RuntimeError(
            "All images appear unreadable. Please validate your dataset files."
        )


def get_transform():
    """
    Get image transformations.

    Returns:
        transform: Composed transformations
    """
    imgnet_mean = [0.485, 0.456, 0.406]
    imgnet_std = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=imgnet_mean, std=imgnet_std),
        ]
    )


def load_imagenet(root, transform=None, val_split=0.2):
    """
    Load ImageNet dataset using torchvision's ImageFolder.

    Args:
        root: Root directory of the ImageNet dataset
        transform: Transformations to apply to images
        val_split: Fraction of data to use for validation

    Returns:
        dict: Dictionary containing full, train, and validation datasets
    """
    dataset = SafeImageFolder(root=root, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    return {"full": dataset, "train": train_set, "val": val_set}


def _transform_batch(batch, transform):
    """
    Apply transformations to a batch of images.

    Args:
        batch: Batch from DataLoader
        transform: Composed transformations to apply

    Returns:
        batch: Transformed batch
    """
    batch["image"] = [transform(image) for image in batch["image"]]
    return batch


def load_hf_dataset(name, transform=None):
    """
    Load huggingface dataset and apply transformations.

    Args:
        name: Name of the dataset to load
        transform: Transformations to apply to images

    Returns:
        dataset: huggingface dataset
    """
    dataset = load_dataset(name)
    dataset = dataset.with_format("torch")
    fn = partial(_transform_batch, transform=transform)
    dataset.set_transform(fn)
    return dataset
