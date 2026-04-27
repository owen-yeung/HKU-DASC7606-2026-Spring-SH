# Training run profiles: smoke vs full

This project uses `config.py` for all training hyperparameters. **At any time, the checked-in `config.py` reflects the profile you intend to run next** (currently the **full** run). This file records both the **smoke** and **full** profiles and why each choice was made.

Hardware reference: **NVIDIA GeForce RTX 4080 (16 GB)**, with a budget of roughly **10 hours** total for quick validation, optional sweeps, and the main fine-tuning run.

---

## Shared settings (both smoke and full)

These are architectural and data choices that do not differ between the two profiles.

| Area | Setting | Rationale |
|------|---------|-----------|
| **Backbone** | `IMAGE_ENCODER = "resnet"` | ResNet50 + projection; stable and memory-friendly on 16 GB compared to larger ViT runs. |
| **Alignment** | `EMBED_DIM = 512` | Projected joint embedding size used by both towers. |
| **Partial fine-tuning** | `UNFREEZE_IMAGE_BLOCKS = 1`, `UNFREEZE_TEXT_LAYERS = 1` | Unfreeze only the top ResNet stage and the last RoBERTa layer so most pretrained weights stay fixed; adapts the model without full finetuning cost and overfit risk. |
| **Data** | `VAL_SPLIT = 0.2`, `IMGNET_DIR` as in config | Standard holdout for `Trainer` validation metrics. |
| **Text (train)** | `TEXT_TEMPLATES` (4 templates) | Random template per sample improves robustness of text embeddings vs a single “a photo of {}.” |
| **Text (eval / predict)** | `EVAL_TEXT_TEMPLATES` (same four) | Prompt ensembling: `topk_evaluate` and `predict.py` average logits over templates, which often helps zero-shot-style classification. |
| **Schedulers (full pipeline)** | `LR_SCHEDULER_TYPE = "cosine"`, `WARMUP_RATIO = 0.05` | Cosine decay with a short warmup is a strong default for fine-tuning; warmup reduces early-step instability when parts of the encoder are trainable. |
| **Objectives** | Single combination in `LR_SWEEP` × `WEIGHT_DECAY_SWEEP` × `TEMPERATURE_SWEEP` | We prioritized **one long run** over a grid search to stay within the 10-hour budget; sweeps are optional later. |

---

## Smoke run (~30 minutes)

**Goal:** Verify that the pipeline runs end-to-end on the cluster GPU: no OOM, reasonable step time, loss moving in the right direction, and one validation epoch completes. Not intended to maximize accuracy.

| Setting | Smoke value | Rationale |
|---------|-------------|-----------|
| `SWEEP_NUM_EPOCHS` / `NUM_EPOCHS` | `1` | One pass through the training split is enough to surface integration issues and rough throughput. |
| `OUTPUT_DIR` | `./clip-finetuned/smoke-1epoch` | Isolated from full-run artifacts. |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | **Faster** steps than accum=2; effective batch 128. Smoke is not optimizing for contrastive batch size. |
| `EVAL_BATCH_SIZE` | `256` | Larger eval batch speeds up the single validation pass. |
| `LOG_STEPS` | `50` | Fewer logging points on a very short run reduces I/O noise; still enough points to see a loss curve. |
| `LR_SWEEP` | `[2.5e-4]` | Same order of magnitude as the final run so behavior is representative (not a toy LR). |
| `WEIGHT_DECAY_SWEEP` | `[0.01]` | Lighter regularization than `0.1` for head + partial-backbone training; matches the intended final run. |
| `TEMPERATURE_SWEEP` / `TEMPERATURE` | `[0.05]`, `0.05` | Slightly sharper logits than 0.07; aligned with earlier small sweeps where 0.05 was competitive. |
| `EVAL_PUSHABLE_DIR` | `./eval-reports/smoke-1epoch` | Keeps smoke eval exports separate from the full run. |

**After smoke:** Update `BEST_MODEL_PATH` to the new `.../smoke-1epoch/<run_id>/checkpoint-*/model.safetensors` if you run `eval.py` / `predict.py` against the smoke checkpoint.

---

## Full run (~10 hours budget)

**Goal:** Main fine-tuning run with a larger **effective batch** (more in-batch negatives for the contrastive loss) and enough epochs to approach convergence under cosine schedule.

| Setting | Full value | Rationale |
|---------|------------|-----------|
| `SWEEP_NUM_EPOCHS` / `NUM_EPOCHS` | `11` | Chosen in the 10–12 epoch range: more than the earlier 3-epoch experiments where loss was still improving; 11 is a single rounded target within the 10h ceiling (adjust to 10 or 12 if wall-clock is tight or loose). |
| `OUTPUT_DIR` | `./clip-finetuned/full-run_lr2.5e-04_wd0.01_epochs11` | Descriptive, avoids overwriting other experiments. |
| `GRADIENT_ACCUMULATION_STEPS` | `2` | **Effective train batch = 128 × 2 = 256** with per-device batch 128, improving contrastive learning without requiring a single 256-width forward (often infeasible on 16 GB for this model). |
| `TRAIN_BATCH_SIZE` | `128` | Per-step batch before accumulation. |
| `EVAL_BATCH_SIZE` | `256` | Faster validation each epoch. |
| `LOG_STEPS` | `10` | Denser training loss logging over a long run for better curves and debugging. |
| `LEARNING_RATE` / `LR_SWEEP` | `2.5e-4` | Slightly conservative vs `3e-4` now that the top of the image/text stacks train and augmentation is on; can still be increased if the learning curve is very flat. |
| `WEIGHT_DECAY` / `WEIGHT_DECAY_SWEEP` | `0.01` | Suited to adapter-style tuning; `0.1` can over-regularize small trainable blocks. |
| `TEMPERATURE` / `TEMPERATURE_SWEEP` | `0.05` | Kept consistent with the chosen sweep point and `eval`/`predict`. |
| `EVAL_PUSHABLE_DIR` | `./eval-reports/full-run` | Central place for post-training eval artifacts. |

**After full training:** Set `BEST_MODEL_PATH` to the best `checkpoint-*` (Trainer uses `load_best_model_at_end` on `accuracy`).

---

## Files that encode behavior beyond `config.py`

- **Train:** `train.py` — `TrainingArguments` use `GRADIENT_ACCUMULATION_STEPS`, scheduler type, and warmup.
- **Data:** `data/dataset.py` — strong **train** transforms vs deterministic **eval** transforms.
- **Eval / predict:** `utils.py` (template ensembling in `topk_evaluate`), `eval.py`, `predict.py`.

---

## Quick reference: switching profiles

1. **Smoke:** Set `SWEEP_NUM_EPOCHS = 1`, `NUM_EPOCHS = 1`, `OUTPUT_DIR` to a smoke path, `GRADIENT_ACCUMULATION_STEPS = 1`, `LOG_STEPS = 50`, and point `EVAL_PUSHABLE_DIR` at a smoke subfolder.
2. **Full:** Use the table above (or copy from the current `config.py` in the repository).

This document is **descriptive**; the single source of truth for the next run is always **`config.py`**.
