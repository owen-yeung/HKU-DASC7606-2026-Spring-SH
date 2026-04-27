# Evaluation and Logging Writeup

This document explains how evaluation is run in `eval.py`, what is logged, and where outputs are saved.

## 1) Evaluation flow

`eval.py` runs zero-shot evaluation on three datasets:

1. ImageNet validation split from local `ImageFolder` data (`Config.IMGNET_DIR`)
2. CIFAR-10 (train + test combined)
3. CIFAR-100 (train + test combined, using `fine_label` as `label`)

For each dataset, the script:

1. Builds/loads the dataset
2. Calls `topk_evaluate(...)` from `utils.py`
3. Saves both full and compact JSON outputs
4. Prints a one-line summary to stdout

## 2) Model weight modes (pretrained vs fine-tuned)

`eval.py` supports CLI flags:

- `--weights {pretrained,finetuned}`  
  - `pretrained`: constructs `CLIP(..., pretrained=True)` and does **not** load a checkpoint
  - `finetuned`: constructs `CLIP(..., pretrained=False)` and loads a checkpoint
- `--checkpoint <path>`  
  - path to `.safetensors` file (used only in `finetuned` mode)
  - defaults to `Config.BEST_MODEL_PATH`

### Example commands

Pretrained baseline:

```bash
python eval.py --weights pretrained
```

Fine-tuned eval with default checkpoint:

```bash
python eval.py --weights finetuned
```

Fine-tuned eval with explicit checkpoint:

```bash
python eval.py --weights finetuned --checkpoint ./clip-finetuned/.../model.safetensors
```

## 3) Output directories

Each run creates timestamped output folders:

- Full results:
  - `Config.OUTPUT_DIR/eval/<YYYY-MM-DD_HH-MM-SS>/`
- Compact/pushable results:
  - `Config.EVAL_PUSHABLE_DIR/<YYYY-MM-DD_HH-MM-SS>/`

Each dataset writes:

- `<dataset>.json` (full)
- `<dataset>.small.json` (compact)

Dataset names currently used:

- `imagenet_val`
- `cifar10_all`
- `cifar100_all`

## 4) What `topk_evaluate` computes

`topk_evaluate(...)` in `utils.py`:

1. Builds text prompts from class names (e.g., `"a photo of {}."`)
2. Runs inference batch-by-batch
3. Collects probabilities and labels for the full dataset
4. Computes requested top-k metrics
5. If `return_details=True`, computes extended diagnostics

### Core metrics

- `metrics` dictionary with keys like:
  - `top1_accuracy`
  - `top5_accuracy`
  - `top10_accuracy`
  - (CIFAR-10 currently uses `top1/top2/top3`)

### Extended diagnostics

- `balanced_accuracy` (mean per-class top-1 accuracy over present classes)
- `calibration`
  - `ece_15bins`
  - `brier_score`
- `confidence_stats`
  - `avg_confidence_correct`
  - `avg_confidence_incorrect`
- `top_confusions`
  - top 20 `(true class -> predicted class)` error pairs with counts
- `per_class_accuracy`
  - one item per class with:
    - `label_id`
    - `label_name`
    - `num_samples`
    - `num_correct_top1`
    - `top1_accuracy`
- `num_samples`
- `num_misclassified`
- `misclassified_samples` (full list of top-1 failures)

## 5) Full JSON schema (current)

Each full result file (`<dataset>.json`) has:

```json
{
  "metrics": {
    "top1_accuracy": 0.0,
    "top5_accuracy": 0.0,
    "top10_accuracy": 0.0
  },
  "balanced_accuracy": 0.0,
  "calibration": {
    "ece_15bins": 0.0,
    "brier_score": 0.0
  },
  "confidence_stats": {
    "avg_confidence_correct": 0.0,
    "avg_confidence_incorrect": 0.0
  },
  "num_samples": 0,
  "num_misclassified": 0,
  "top_confusions": [
    {
      "true_label_id": 0,
      "true_label_name": "class_a",
      "predicted_label_id": 1,
      "predicted_label_name": "class_b",
      "count": 0
    }
  ],
  "per_class_accuracy": [
    {
      "label_id": 0,
      "label_name": "class_a",
      "num_samples": 0,
      "num_correct_top1": 0,
      "top1_accuracy": 0.0
    }
  ],
  "misclassified_samples": [
    {
      "sample_index": 0,
      "true_label_id": 0,
      "true_label_name": "class_a",
      "predicted_label_id": 1,
      "predicted_label_name": "class_b",
      "predicted_confidence": 0.0
    }
  ]
}
```

## 6) Compact JSON schema

Each compact file (`<dataset>.small.json`) contains:

- `dataset`
- `metrics`
- `num_samples`
- `num_misclassified`
- `top_n` (from `Config.EVAL_PUSHABLE_TOP_N`)
- `misclassified_samples_top_n` (first `top_n` entries)

This file is intentionally lighter for sharing/log pushing.

## 7) Notes and interpretation tips

- Compare `pretrained` vs `finetuned` runs on the same datasets for a fair baseline/improvement report.
- Use `balanced_accuracy` and `per_class_accuracy` to detect class-wise regressions hidden by overall top-1.
- Use calibration metrics (`ece_15bins`, `brier_score`) to understand confidence quality, not just correctness.
- `top_confusions` is useful for targeted error analysis and prompt-template refinement.
