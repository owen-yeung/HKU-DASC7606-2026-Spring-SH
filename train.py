import os
from datetime import datetime
from functools import partial
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer
from config import Config
from data.dataset import get_transform, load_imagenet
from utils import clip_data_collator, compute_metrics
from model.clip import CLIP

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
    text_templates=Config.TEXT_TEMPLATES,
)

for learning_rate in Config.LR_SWEEP:
    run_name = f"lr_{learning_rate:.0e}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    output_dir = os.path.join(Config.OUTPUT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Starting run: {run_name} ===")

    model = CLIP(
        encoder_type=Config.IMAGE_ENCODER,
        embed_dim=Config.EMBED_DIM,
        temperature=Config.TEMPERATURE,
        pretrained=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=Config.SWEEP_NUM_EPOCHS,
        learning_rate=learning_rate,
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
    eval_metrics = trainer.evaluate()
    print(f"Validation metrics for lr={learning_rate}: {eval_metrics}")

    # Plot and save training loss curve from Trainer logs.
    train_steps = []
    train_losses = []
    for log_entry in trainer.state.log_history:
        if "loss" in log_entry and "eval_loss" not in log_entry:
            train_steps.append(log_entry.get("step", len(train_steps) + 1))
            train_losses.append(log_entry["loss"])

    if train_losses:
        def save_loss_curve(steps, losses, title, filename):
            plt.figure(figsize=(8, 5))
            plt.plot(steps, losses, marker="o", linewidth=1.5, markersize=3)
            plt.title(title)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            curve_path = os.path.join(output_dir, filename)
            plt.savefig(curve_path, dpi=150)
            plt.close()
            return curve_path

        loss_curve_path = save_loss_curve(
            train_steps,
            train_losses,
            f"Training Loss Curve (lr={learning_rate})",
            "training_loss_curve.png",
        )
        print(f"Saved training loss curve to: {loss_curve_path}")

        post_first_epoch_steps = []
        post_first_epoch_losses = []
        for log_entry in trainer.state.log_history:
            if (
                "loss" in log_entry
                and "eval_loss" not in log_entry
                and log_entry.get("epoch") is not None
                and log_entry["epoch"] > 1
            ):
                post_first_epoch_steps.append(
                    log_entry.get("step", len(post_first_epoch_steps) + 1)
                )
                post_first_epoch_losses.append(log_entry["loss"])

        if post_first_epoch_losses:
            post_first_epoch_curve_path = save_loss_curve(
                post_first_epoch_steps,
                post_first_epoch_losses,
                f"Training Loss Curve After Epoch 1 (lr={learning_rate})",
                "training_loss_curve_after_first_epoch.png",
            )
            print(
                "Saved post-first-epoch training loss curve to: "
                f"{post_first_epoch_curve_path}"
            )
        else:
            print(
                "No post-first-epoch training loss entries found in trainer log "
                f"history for lr={learning_rate}."
            )
    else:
        print(
            f"No training loss entries found in trainer log history for lr={learning_rate}."
        )