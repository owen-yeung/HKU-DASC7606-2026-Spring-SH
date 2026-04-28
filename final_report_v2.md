# Final Report: CLIP Fine-Tuning and Ablations

## 1. Methods

### 1.1. Architecture choices

- **Vision encoder:** `ResNet50` (`IMAGE_ENCODER="resnet"` in `config.py`) for stable training and manageable memory usage on 16 GB GPU.
- **Text encoder:** `RoBERTa-base` (`model/text_encoder.py`) with masked mean pooling over token embeddings.
- **Shared embedding space:** image/text embeddings are projected to `EMBED_DIM=512`.
- **Projection heads:** MLP heads (`Linear -> ReLU -> Linear`) on both towers.
- **Similarity + objective:** L2-normalized embeddings with learnable temperature (`log_temperature`), trained using symmetric contrastive loss (image-to-text CE + text-to-image CE, averaged).

### 1.2. Prompt Ablation experiments

- **A->C ladder:**
  - **Exp A:** baseline templates
  - **Exp B:** expanded prompt pack
  - **Exp C:** class-aware prompt variants
- **Ablated dimensions:** prompt template set and class-specific aliases.
- **Selection metric:** weighted top-1 score in `eval.py` (default: CIFAR-10 `0.4`, CIFAR-100 `0.4`, ImageNet `0.2`; auto re-normalized when ImageNet is skipped).

## 2. Fine-tuning

### 2.1. Hyperparameters, run configs

- **Fine-tuning strategy:** freeze most pretrained weights; unfreeze only top layers (`UNFREEZE_IMAGE_BLOCKS=1`, `UNFREEZE_TEXT_LAYERS=1`).
- **Optimizer setup:** LR `2.5e-4`, weight decay `0.01`, temperature `0.05`.
- **Schedule:** cosine decay with warmup ratio `0.05`.
- **Training length:** `11` epochs.
- **Batching:** train batch `128` with grad accumulation `2` (effective `256`), eval batch `256`.
- **Validation snapshot (epoch 11):** eval loss `0.3245`; top-1/top-5/top-10 `0.7873/0.9997/0.9999`; best checkpoint `checkpoint-6256`.

### 2.2. Training loss curves

Training loss curveTraining loss curve after epoch 1

- Training is stable: sharp early loss drop, then gradual improvement with moderate stochastic noise.
- No divergence is observed across the 11-epoch run.

## 3. Prompting Ablation experiments

### 3.1. Overview + Motivation

- ImageNet performance near 100% for fine-tuned model, but CIFAR performance is far behind
- This suggests issues with prompting strategy rather than fine-tuning
- So I shifted GPU resources to running prompting ablation experiments instead of hyperparameter-tuning experiments.
- Experiments ladder:
  - Exp A: baseline templates
  - Exp B: expanded prompt pack
  - Exp C: class-aware prompt variants
- Ablated dimensions: prompt template set and class-specific aliases.
- Selection metric: weighted top-1 score in `eval.py` (default: CIFAR-10 `0.4`, CIFAR-100 `0.4`, ImageNet `0.2`; auto re-normalized when ImageNet is skipped).

### 3.2. ImageNet performance

- ImageNet was skipped in ablation scoring to reduce runtime and prioritize CIFAR-focused iteration.
- This trade-off is reasonable because final no-extra-ablation ImageNet performance is already near saturation (top-1 about `0.9986`).
- **Final fine-tuned model:** top-1 `0.9986`, top-5 `0.9997`, top-10 `0.9999`.

Pretrained weights (no fine-tuning) baseline:

- **Pretrained model:** top-1 `0.0013`, top-5 `0.0078`, top-10 `0.0178`.

Fine-tuning is therefore the dominant driver of ImageNet accuracy gains.

### 3.3. CIFAR performance across ablation sweeps (same table as before)

- Final choice: prompt-pack recipe without extra TTA/weighting (B/C tie; simpler config preferred).
- This is the prompt setup used for the final predictions.json generation


| Setup                                   | CIFAR-10 Top-1 | CIFAR-100 Top-1 | CIFAR-10 Top-2 / Top-3 | CIFAR-100 Top-5 / Top-10 |
| --------------------------------------- | -------------- | --------------- | ---------------------- | ------------------------ |
| Pretrained weights, no prompt-pack      | 0.1462         | 0.0103          | 0.2770 / 0.3971        | 0.0561 / 0.1091          |
| Pretrained weights, with prompt-pack    | 0.1370         | 0.0078          | 0.2449 / 0.3511        | 0.0455 / 0.0957          |
| Final weights, Exp A (no prompt-pack)   | 0.4290         | 0.1161          | 0.6110 / 0.7180        | 0.2778 / 0.3905          |
| Final weights, Exp B (with prompt-pack) | 0.4379         | 0.1208          | 0.6226 / 0.7304        | 0.2853 / 0.3988          |
| Final weights, Exp C                    | 0.4379         | 0.1208          | 0.6226 / 0.7304        | 0.2853 / 0.3988          |


## 4. Lingering weaknesses analysis

### 4.1 Recurring failure patterns (post-eval artifacts)

Analysis of misclassifed images shows consistent confusion pairs across ablation setups:

- **CIFAR-10:**
  - `automobile -> truck`
  - `airplane -> ship`
  - `cat <-> dog`
  - `horse -> deer`, `horse -> dog`
  - `bird -> dog`, `bird -> cat`
  - `frog -> deer`, `frog -> dog`
- **CIFAR-100:**
  - `keyboard -> clock`
  - `orange -> rose`
  - `man -> chimpanzee`
  - `worm -> snake`
  - `shark -> sea`
  - `streetcar -> house`

The same over-predicted wrong labels also recur: **dog/deer/truck/ship** on CIFAR-10 and **sweet_pepper/porcupine/chimpanzee/otter/house** on CIFAR-100.

### 4.2 Why these errors persist

- Most failure cases are tiny (`32x32`) and visually ambiguous after resize/normalization.
- Errors are concentrated in semantically similar groups (vehicle-vehicle, mammal-mammal, object-object), indicating limited fine-grained discrimination under low resolution.
- Prompting improvements help overall metrics but do not materially change dominant confusion structure, especially on CIFAR-100.

### 4.3 Illustrative misclassification examples

**CIFAR-10 examples**

- True: `airplane`, Pred: `ship`  
CIFAR-10 airplane predicted as ship
- True: `bird`, Pred: `deer`  
CIFAR-10 bird predicted as deer
- True: `horse`, Pred: `dog`  
CIFAR-10 horse predicted as dog
- True: `automobile`, Pred: `truck`  
CIFAR-10 automobile predicted as truck

**CIFAR-100 examples**

- True: `cattle`, Pred: `boy`  
CIFAR-100 cattle predicted as boy
- True: `dinosaur`, Pred: `snake`  
CIFAR-100 dinosaur predicted as snake
- True: `cloud`, Pred: `lamp`  
CIFAR-100 cloud predicted as lamp
- True: `keyboard`, Pred: `clock`  
CIFAR-100 keyboard predicted as clock

