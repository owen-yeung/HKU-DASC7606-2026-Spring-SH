import os
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer
from config import Config
from data.dataset import get_transform, load_imagenet
from utils import clip_data_collator, compute_metrics
from model.clip import CLIP

# CELL 2
output_dir = f"{Config.OUTPUT_DIR}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
os.makedirs(output_dir, exist_ok=True)

transform = get_transform()
datasets = load_imagenet(
    root=Config.IMGNET_DIR,
    transform=transform,
    val_split=Config.VAL_SPLIT,
)
class_names = datasets["full"].classes

data_collator = partial(
    clip_data_collator,
    class_names=class_names,
    text_template=Config.TEXT_TEMPLATE,
)

model = CLIP(
    encoder_type=Config.IMAGE_ENCODER,
    embed_dim=Config.EMBED_DIM,
    temperature=Config.TEMPERATURE,
    pretrained=True,
)

# CELL 3
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=Config.NUM_EPOCHS,
    learning_rate=Config.LEARNING_RATE,
    per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
    weight_decay=Config.WEIGHT_DECAY,
    logging_steps=Config.LOG_STEPS,
    dataloader_num_workers=Config.NUM_WORKERS,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["val"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Plot and save training loss curve from Trainer logs.
train_steps = []
train_losses = []
for log_entry in trainer.state.log_history:
    if "loss" in log_entry and "eval_loss" not in log_entry:
        train_steps.append(log_entry.get("step", len(train_steps) + 1))
        train_losses.append(log_entry["loss"])

if train_losses:
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_losses, marker="o", linewidth=1.5, markersize=3)
    plt.title("Training Loss Curve")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    loss_curve_path = os.path.join(output_dir, "training_loss_curve.png")
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()
    print(f"Saved training loss curve to: {loss_curve_path}")
else:
    print("No training loss entries found in trainer log history.")