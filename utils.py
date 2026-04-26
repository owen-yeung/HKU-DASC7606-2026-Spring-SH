import random
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def clip_data_collator(batch, class_names, text_templates):
    """
    Custom data collator for CLIP training.
    Converts a batch of (image, label_id) pairs into a format suitable for CLIP.

    Args:
        batch: List of tuples (image, label_id)
        class_names: List of class names
        text_templates: List of templates
    Returns:
        A dictionary with keys "images", "texts", and "labels" for CLIP training
    """
    # TODO: Iterate over batch of (image, label_id) pairs
    # Format text prompts using class_names and text_templates
    images = []
    texts = []
    for image, label_id in batch:
        images.append(image)
        class_name = class_names[int(label_id)]
        template = random.choice(text_templates)
        texts.append(template.format(class_name))

    # TODO: Stack images into a single tensor [batch_size, C, H, W]
    images = torch.stack(images)

    # TODO: Labels: Diagonal elements are positive pairs (correct image-text matches)
    labels = torch.arange(len(batch), dtype=torch.long)

    return {"images": images, "texts": texts, "labels": labels}


def compute_metrics(eval_pred):
    """
    Compute top-1, top-5, top-10 accuracy for CLIP evaluation.
    Aggregate outputs from the entire evaluation dataset.
    Args:
        eval_pred: Tuple (logits, labels) where:
            - logits: (num_samples, num_classes) similarity scores from CLIP
            - labels: (num_samples,) correct match indices (diagonal labels)
    Returns:
        Dictionary with "accuracy", "top5_accuracy", and "top10_accuracy"
    """
    # TODO: Unpack logits and labels from eval_pred
    logits, labels = eval_pred
    logits = torch.as_tensor(logits)
    labels = torch.as_tensor(labels).long()
    num_classes = logits.shape[1]

    # TODO: Get indices of top-k [1, 5, 10] predictions
    # Check if the correct label appears in the top-k predictions
    # Compute top-k accuracy
    max_k = min(10, num_classes)
    topk_indices = logits.topk(k=max_k, dim=1).indices
    labels_expanded = labels.unsqueeze(1)
    matches = topk_indices.eq(labels_expanded)

    accuracy = matches[:, :1].any(dim=1).float().mean().item()
    top5_k = min(5, num_classes)
    top5_accuracy = matches[:, :top5_k].any(dim=1).float().mean().item()
    top10_accuracy = matches[:, :max_k].any(dim=1).float().mean().item()

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_accuracy,
        "top10_accuracy": top10_accuracy,
    }


def topk_evaluate(
    model,
    dataset,
    class_names,
    batch_size=64,
    num_workers=4,
    text_template="a photo of {}",
    topk=[1, 5, 10],
):
    """
    Evaluate CLIP model on a dataset and compute top-k accuracy.
    Args:
        model: CLIP model to evaluate
        dataset: Dataset to evaluate on (should return dict with "image" and "label", or tuple of (image, label))
        class_names: List of class names corresponding to label IDs
        batch_size: Batch size for evaluation
        num_workers: Number of workers for DataLoader
        text_template: Template for generating text prompts
        topk: List of k values for top-k accuracy (e.g. [1, 5, 10])
    """
    # TODO: Create dataloader and generate text prompts
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    text_prompts = [text_template.format(class_name) for class_name in class_names]
    device = next(model.parameters()).device

    # TODO: Get all predictions, probabilities, and true labels
    model.eval()
    all_probs = []
    all_labels = []
    max_k = min(max(topk), len(class_names))
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if isinstance(batch, dict):
                images = batch["image"]
                labels = batch["label"]
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device).long()
            _, probabilities = model.predict(images, text_prompts)
            all_probs.append(probabilities.cpu())
            all_labels.append(labels.cpu())

    # TODO: Get top-k indices, [N, k]
    # Compute top-k accuracy: fraction where true label appears in top-k indices
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    topk_indices = all_probs.topk(k=max_k, dim=1).indices
    matches = topk_indices.eq(all_labels.unsqueeze(1))

    metrics = {}
    for k in topk:
        k_eff = min(k, len(class_names))
        metrics[f"top{k}_accuracy"] = matches[:, :k_eff].any(dim=1).float().mean().item()

    return metrics
