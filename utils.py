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
    text_templates=None,
    topk=[1, 5, 10],
    return_details=False,
):
    """
    Evaluate CLIP model on a dataset and compute top-k accuracy.
    Args:
        model: CLIP model to evaluate
        dataset: Dataset to evaluate on (should return dict with "image" and "label", or tuple of (image, label))
        class_names: List of class names corresponding to label IDs
        batch_size: Batch size for evaluation
        num_workers: Number of workers for DataLoader
        text_template: Legacy single template for generating text prompts
        text_templates: Optional list of templates for prompt ensembling
        topk: List of k values for top-k accuracy (e.g. [1, 5, 10])
        return_details: If True, also return top-1 misclassified samples
    """
    # TODO: Create dataloader and generate text prompts
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    effective_templates = text_templates if text_templates else [text_template]
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
            logits_per_template = []
            for template in effective_templates:
                text_prompts = [template.format(class_name) for class_name in class_names]
                logits = model.compute_similarity(images, text_prompts)
                logits_per_template.append(logits)
            ensemble_logits = torch.stack(logits_per_template, dim=0).mean(dim=0)
            probabilities = torch.softmax(ensemble_logits, dim=1)
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

    if not return_details:
        return metrics

    top1_probs, top1_preds = all_probs.max(dim=1)
    top1_correct = top1_preds.eq(all_labels)
    failure_mask = top1_preds.ne(all_labels)
    failure_indices = failure_mask.nonzero(as_tuple=True)[0].tolist()

    # Per-class stats for balanced accuracy and class-wise diagnostics.
    num_classes = len(class_names)
    class_counts = torch.bincount(all_labels, minlength=num_classes)
    class_correct = torch.bincount(
        all_labels[top1_correct], minlength=num_classes
    )
    class_accuracy = torch.zeros(num_classes, dtype=torch.float32)
    valid_mask = class_counts > 0
    class_accuracy[valid_mask] = (
        class_correct[valid_mask].float() / class_counts[valid_mask].float()
    )
    balanced_accuracy = (
        class_accuracy[valid_mask].mean().item() if valid_mask.any() else 0.0
    )

    # Confidence diagnostics.
    if top1_correct.any():
        avg_conf_correct = top1_probs[top1_correct].mean().item()
    else:
        avg_conf_correct = 0.0
    if failure_mask.any():
        avg_conf_incorrect = top1_probs[failure_mask].mean().item()
    else:
        avg_conf_incorrect = 0.0

    # Calibration diagnostics: ECE (15 bins) and Brier score.
    ece_bins = 15
    ece = 0.0
    for i in range(ece_bins):
        left = i / ece_bins
        right = (i + 1) / ece_bins
        if i == ece_bins - 1:
            bin_mask = (top1_probs >= left) & (top1_probs <= right)
        else:
            bin_mask = (top1_probs >= left) & (top1_probs < right)
        if not bin_mask.any():
            continue
        bin_acc = top1_correct[bin_mask].float().mean().item()
        bin_conf = top1_probs[bin_mask].mean().item()
        bin_weight = bin_mask.float().mean().item()
        ece += abs(bin_acc - bin_conf) * bin_weight

    one_hot_labels = torch.nn.functional.one_hot(
        all_labels, num_classes=num_classes
    ).float()
    brier_score = ((all_probs - one_hot_labels) ** 2).sum(dim=1).mean().item()

    # Most common confusion pairs: true class -> predicted class for wrong predictions.
    confusion_counts = {}
    for true_label_id, pred_label_id in zip(
        all_labels[failure_mask].tolist(),
        top1_preds[failure_mask].tolist(),
    ):
        key = (int(true_label_id), int(pred_label_id))
        confusion_counts[key] = confusion_counts.get(key, 0) + 1

    top_confusions = sorted(
        confusion_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:20]
    top_confusions = [
        {
            "true_label_id": true_id,
            "true_label_name": class_names[true_id],
            "predicted_label_id": pred_id,
            "predicted_label_name": class_names[pred_id],
            "count": count,
        }
        for (true_id, pred_id), count in top_confusions
    ]

    per_class_accuracy = [
        {
            "label_id": idx,
            "label_name": class_names[idx],
            "num_samples": int(class_counts[idx].item()),
            "num_correct_top1": int(class_correct[idx].item()),
            "top1_accuracy": float(class_accuracy[idx].item()),
        }
        for idx in range(num_classes)
        if class_counts[idx].item() > 0
    ]

    failures = []
    for idx in failure_indices:
        true_label_id = int(all_labels[idx].item())
        pred_label_id = int(top1_preds[idx].item())
        failures.append(
            {
                "sample_index": int(idx),
                "true_label_id": true_label_id,
                "true_label_name": class_names[true_label_id],
                "predicted_label_id": pred_label_id,
                "predicted_label_name": class_names[pred_label_id],
                "predicted_confidence": float(top1_probs[idx].item()),
            }
        )

    return {
        "metrics": metrics,
        "balanced_accuracy": float(balanced_accuracy),
        "calibration": {
            "ece_15bins": float(ece),
            "brier_score": float(brier_score),
        },
        "confidence_stats": {
            "avg_confidence_correct": float(avg_conf_correct),
            "avg_confidence_incorrect": float(avg_conf_incorrect),
        },
        "num_samples": int(all_labels.shape[0]),
        "num_misclassified": int(len(failures)),
        "top_confusions": top_confusions,
        "per_class_accuracy": per_class_accuracy,
        "misclassified_samples": failures,
    }
