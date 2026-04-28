# Final Report: CLIP Fine-Tuning and Evaluation

## 1) Design Choices Overview

This section summarizes the major design decisions made in this project and the rationale behind each one.

### 1.1 CLIP Model Architecture Choices

- **Vision backbone:** `ResNet50` (`IMAGE_ENCODER="resnet"` in `config.py`) was selected as the default image encoder for stability and memory efficiency on a 16 GB RTX 4080.
- **Text backbone:** `RoBERTa-base` is used as the text encoder (`model/text_encoder.py`) instead of CLIP's original text tower, with masked mean pooling over token embeddings.
- **Shared embedding space:** Both image and text towers are projected to `EMBED_DIM=512`.
- **Projection heads:** Both towers use MLP heads (`Linear -> ReLU -> Linear`) rather than a single linear projection.
- **Similarity + loss:** `model/clip.py` computes cosine-style similarities by L2-normalizing both embeddings, applies learnable temperature scaling (`log_temperature`), and optimizes symmetric contrastive loss:
  - image-to-text CE + text-to-image CE, averaged.

### 1.2 Fine-Tuning Strategy

- **Parameter-efficient adaptation:** Most pretrained parameters are frozen. Only top layers are unfrozen:
  - `UNFREEZE_IMAGE_BLOCKS=1`
  - `UNFREEZE_TEXT_LAYERS=1`
- **Training schedule:** cosine LR decay with warmup (`LR_SCHEDULER_TYPE="cosine"`, `WARMUP_RATIO=0.05`).
- **Main run hyperparameters (current full run profile):**
  - LR `2.5e-4`, WD `0.01`, temperature `0.05`
  - Epochs `11`
  - Train batch `128` with grad accumulation `2` (effective batch `256`)
  - Eval batch `256`
- **Data transforms:** stronger randomized train transforms and deterministic eval transforms in `data/dataset.py`.
- **Text augmentation during training:** multiple templates in `TEXT_TEMPLATES` are sampled per batch in `clip_data_collator`.

### 1.3 Ablation Experiments

- **A->E ablation ladder** was implemented to isolate incremental gains:
  - **Exp A:** baseline templates
  - **Exp B:** expanded prompt pack
  - **Exp C:** class-aware prompt variants
  - **Exp D:** add deterministic TTA
  - **Exp E:** add template weighting
- **Recipe dimensions being ablated:**
  - prompt template set
  - class-specific aliases
  - template weights
  - TTA views (`base`, `hflip`, `center_zoom_90`, `center_zoom_80`)
- **Selection criterion:** weighted score in `eval.py` based on top-1 metrics:
  - default weights: CIFAR-10 `0.4`, CIFAR-100 `0.4`, ImageNet `0.2`
  - if ImageNet is skipped, score is automatically re-normalized over available metrics.

### 1.4 Other Important Choices

- **Consistency between prediction and evaluation:**
  - classification/logit generation is centralized in `utils.py` (`compute_ensemble_logits`) and used by both `predict.py` and `eval.py`.
- **Evaluation reporting-first workflow:**
  - sweep outputs include machine-readable summaries and external-report-friendly artifacts.
- **Checkpoint-series evaluation:**
  - `eval.py --run-checkpoint-series` evaluates all `checkpoint-*` models using the best ablation recipe from an ablation summary, enabling training-time performance trajectory analysis.
- **Runtime flexibility:**
  - `--skip-imagenet` allows faster CIFAR-only sweeps/series.

## 2) Current Training and Evaluation Snapshot

### 2.1 Current Fine-Tuned Run

From `clip-finetuned/full-run_lr2.5e-04_wd0.01_epochs11/lr_3e-04_wd_0.01_temp_0.05_2026-04-28_00-04-34/run_summary.json`:

- Validation accuracy: `0.7873`
- Validation top-5: `0.9997`
- Validation top-10: `0.9999`
- Best checkpoint: `checkpoint-6256`
- Training artifacts:
  - `training_loss_curve.png`
  - `training_loss_curve_after_first_epoch.png`

### 2.2 Latest Multi-Dataset Eval Snapshot

From `eval-reports/full-run/2026-04-28_11-37-00`:

- **ImageNet val:** top-1 `0.9986`, top-5 `0.9997`, top-10 `0.9999`
- **CIFAR-10 (train+test):** top-1 `0.4290`, top-2 `0.6110`, top-3 `0.7180`
- **CIFAR-100 (train+test):** top-1 `0.1161`, top-5 `0.2778`, top-10 `0.3905`

Interpretation: the model is very strong on ImageNet-style evaluation but still weak on CIFAR transfer, motivating prompt/TTA/recipe ablations and checkpoint-series analysis.

## 3) Evaluation and Reporting Pipeline

### 3.1 Eval Artifacts Per Dataset

`eval.py` writes, for each dataset:

- full JSON metrics/details
- pushable `.small.json`
- misclassification review `.html`
- sample preview images (`.sample_images/*.png`)

### 3.2 Ablation Reporting Bundle

`eval.py --run-ablations` produces:

- `eval_ablation_summary.json`
- `eval_ablation_leaderboard.csv`
- `eval_ablation_report.md`
- `eval_ablation_report.html`
- `eval_ablation_scores.png` (if matplotlib is available)

### 3.3 Checkpoint-Series Reporting Bundle

`eval.py --run-checkpoint-series` produces:

- `checkpoint_series_summary.json`
- `checkpoint_series_leaderboard.csv`
- `checkpoint_series_report.md`
- `checkpoint_series_report.html`
- `checkpoint_series_scores.png` (if matplotlib is available)

All above artifacts are configured to be pushable via `.gitignore` allow rules.

## 4) Reproducible Commands

- **Run ablation sweep (full):**
  - `python eval.py --run-ablations`
- **Run ablation sweep (faster, CIFAR-only):**
  - `python eval.py --run-ablations --skip-imagenet`
- **Checkpoint-series with best ablation recipe:**
  - `python eval.py --run-checkpoint-series --ablation-summary "<path/to/eval_ablation_summary.json>"`
- **Checkpoint-series CIFAR-only:**
  - `python eval.py --run-checkpoint-series --skip-imagenet --ablation-summary "<path/to/eval_ablation_summary.json>"`

## 5) Key Takeaways and Next Steps

- The architecture and fine-tuning setup are stable and performant on ImageNet-style data.
- Cross-domain transfer (especially CIFAR-100) remains the main bottleneck.
- The implemented ablation + reporting framework now supports:
  - selecting stronger inference recipes,
  - measuring how performance evolves by checkpoint,
  - producing external-report-ready evidence.
- Recommended next step: run full CIFAR-focused ablations and checkpoint-series first (`--skip-imagenet`) to shorten iteration time, then confirm best recipe on ImageNet as a final guardrail.
