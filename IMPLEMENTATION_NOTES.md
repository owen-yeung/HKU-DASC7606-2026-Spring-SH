# CLIP Implementation Notes

This document describes what was implemented in this project and highlights deviations from the original baseline CLIP design.

## Implemented Components

### 1) Core CLIP model (`model/clip.py`)

- Added image encoder initialization via `ImageEncoder` with configurable backbone:
  - `encoder_type="resnet"` -> ResNet-50
  - `encoder_type="vit"` -> ViT-B/16
- Added text pipeline initialization:
  - `TextTokenizer` (RoBERTa tokenizer)
  - `TextEncoder` (RoBERTa backbone + projection MLP)
- Implemented learnable log-temperature:
  - `self.log_temperature = nn.Parameter(torch.tensor(temperature).log())`
  - `temperature` property returns `exp(log_temperature)`
- Implemented image-text similarity computation:
  - Encode image embeddings and text embeddings
  - L2-normalize both feature sets
  - Compute pairwise similarity matrix `logits = image_emb @ text_emb.T`
  - Scale by inverse temperature using division by `temperature`
- Implemented symmetric contrastive objective:
  - image-to-text cross entropy: `CE(logits, labels)`
  - text-to-image cross entropy: `CE(logits.T, labels)`
  - final loss: average of both directions
- Implemented inference helper:
  - `predict(images, texts)` returns `(predictions, probabilities)`
  - `probabilities = softmax(logits, dim=1)`

### 2) Image encoder stack (`model/image_encoder.py`)

- **ResNet50 encoder**
  - Loads ImageNet-pretrained ResNet-50 (optional)
  - Captures feature dimension from classifier input (`2048`)
  - Removes classifier head and keeps convolutional feature extractor
  - Flattens output from `[B, 2048, 1, 1]` to `[B, 2048]`
- **ViT-B/16 encoder**
  - Loads ImageNet-pretrained ViT-B/16 (optional)
  - Captures feature dimension from head input (`768`)
  - Replaces classification head with identity
  - Uses model output feature vector (CLS-based representation in torchvision ViT forward path)
- **ImageEncoder wrapper**
  - Supports backbone selection (`resnet` or `vit`)
  - Freezes all pretrained backbone parameters
  - Adds trainable projection MLP:
    - `Linear(feature_dim, feature_dim) -> ReLU -> Linear(feature_dim, embed_dim)`
  - Outputs projected image embeddings

### 3) Text encoder stack (`model/text_encoder.py`)

- Loads RoBERTa-base with `add_pooling_layer=False` (pretrained or config-based init)
- Freezes RoBERTa backbone parameters
- Adds trainable projection MLP:
  - `Linear(hidden_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, embed_dim)`
- Implements masked mean pooling:
  - Uses `attention_mask` to ignore padding tokens
  - Computes sentence embedding by averaging valid token embeddings
- Projects pooled sentence embeddings to shared CLIP embedding space
- `TextTokenizer` uses RoBERTa tokenizer and returns `input_ids` and `attention_mask` on target device

### 4) Data and evaluation utilities (`utils.py`)

- Implemented `clip_data_collator(...)`:
  - Converts batch `(image, label_id)` into:
    - stacked image tensor
    - text prompts generated from random template and class name
    - diagonal-match labels `0..B-1`
- Implemented `compute_metrics(eval_pred)`:
  - Computes top-1, top-5, top-10 accuracy from logits/labels
  - Handles class count smaller than 5/10 safely
- Implemented `topk_evaluate(...)`:
  - Builds DataLoader
  - Generates class prompts using template
  - Calls `model.predict(...)` batch-wise
  - Computes requested top-k accuracies over full dataset

---

## Modifications vs Standard Baseline CLIP

Compared with the original CLIP formulation from OpenAI, this implementation includes several practical simplifications and substitutions:

1. **Text backbone changed from Transformer CLIP text encoder to RoBERTa**
   - Baseline CLIP uses a custom jointly-trained Transformer text tower.
   - Here, RoBERTa-base is used as a frozen language backbone.

2. **Backbones are frozen; only projection heads and temperature are trainable**
   - Baseline CLIP typically trains both vision and text towers end-to-end at scale.
   - In this implementation, pretrained vision/text backbones are frozen by design.

3. **Projection head depth differs from common CLIP linear projections**
   - Baseline CLIP commonly uses a single learned linear projection per tower.
   - Here, each tower uses an MLP projection:
     - `Linear -> ReLU -> Linear`

4. **Text pooling strategy uses masked mean pooling**
   - Baseline CLIP text representation is based on a specific token position from CLIP text transformer.
   - Here, sentence features are computed via attention-mask-aware mean pooling over all non-padding tokens.

5. **Temperature parameterization uses learnable log-temperature without explicit clamp**
   - Baseline CLIP implementations often clamp or constrain logit scaling for stability.
   - This version learns log-temperature directly and exponentiates at runtime.

6. **Prompting and supervision are class-template based**
   - Training collator builds text prompts from class names and random templates.
   - Positive supervision is diagonal pairing between batched images and generated texts.

---

## Notes

- `# TODO` comments were intentionally kept in code (as requested) even where implementations are complete.
- The project setup may still show IDE import-resolution warnings in environments where `torch`, `torchvision`, `transformers`, or `tqdm` are not installed/interpreted by the active Python environment.
