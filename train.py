import os
import json
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

best_eval_accuracy: float | None = None
best_run_info: dict[str, float | str] | None = None

for learning_rate in Config.LR_SWEEP:
    for weight_decay in Config.WEIGHT_DECAY_SWEEP:
        for temperature in Config.TEMPERATURE_SWEEP:
            run_name = (
                f"lr_{learning_rate:.0e}_wd_{weight_decay:g}_temp_{temperature:g}_"
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            )
            output_dir = os.path.join(Config.OUTPUT_DIR, run_name)
            os.makedirs(output_dir, exist_ok=True)
            print(f"\n=== Starting run: {run_name} ===")

            model = CLIP(
                encoder_type=Config.IMAGE_ENCODER,
                embed_dim=Config.EMBED_DIM,
                temperature=temperature,
                pretrained=True,
            )

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=Config.SWEEP_NUM_EPOCHS,
                learning_rate=learning_rate,
                per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
                weight_decay=weight_decay,
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
            print(
                "Validation metrics for "
                f"lr={learning_rate}, wd={weight_decay}, temp={temperature}: "
                f"{eval_metrics}"
            )
            loss_curve_path = None
            post_first_epoch_curve_path = None

            # Plot and save training loss curve from Trainer logs.
            train_steps: list[int] = []
            train_losses: list[float] = []
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
                    "Training Loss Curve "
                    f"(lr={learning_rate}, wd={weight_decay}, temp={temperature})",
                    "training_loss_curve.png",
                )
                print(f"Saved training loss curve to: {loss_curve_path}")

                post_first_epoch_steps: list[int] = []
                post_first_epoch_losses: list[float] = []
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
                        "Training Loss Curve After Epoch 1 "
                        f"(lr={learning_rate}, wd={weight_decay}, temp={temperature})",
                        "training_loss_curve_after_first_epoch.png",
                    )
                    print(
                        "Saved post-first-epoch training loss curve to: "
                        f"{post_first_epoch_curve_path}"
                    )
                else:
                    print(
                        "No post-first-epoch training loss entries found in trainer log "
                        "history for "
                        f"lr={learning_rate}, wd={weight_decay}, temp={temperature}."
                    )
            else:
                print(
                    "No training loss entries found in trainer log history for "
                    f"lr={learning_rate}, wd={weight_decay}, temp={temperature}."
                )

            run_summary = {
                "run_name": run_name,
                "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "temperature": temperature,
                "num_epochs": Config.SWEEP_NUM_EPOCHS,
                "train_batch_size": Config.TRAIN_BATCH_SIZE,
                "eval_batch_size": Config.EVAL_BATCH_SIZE,
                "eval_metrics": eval_metrics,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "training_loss_curve": (
                    os.path.basename(loss_curve_path) if loss_curve_path else None
                ),
                "training_loss_curve_after_first_epoch": (
                    os.path.basename(post_first_epoch_curve_path)
                    if post_first_epoch_curve_path
                    else None
                ),
            }
            run_summary_path = os.path.join(output_dir, "run_summary.json")
            with open(run_summary_path, "w", encoding="utf-8") as summary_file:
                json.dump(run_summary, summary_file, indent=2)
            print(f"Saved run summary to: {run_summary_path}")

            run_eval_accuracy = eval_metrics.get("eval_accuracy")
            if run_eval_accuracy is not None:
                run_eval_accuracy = float(run_eval_accuracy)
                if (
                    best_eval_accuracy is None
                    or run_eval_accuracy > best_eval_accuracy
                ):
                    best_eval_accuracy = run_eval_accuracy
                    best_run_info = {
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "temperature": temperature,
                    }

if best_run_info is not None and best_eval_accuracy is not None:
    print("\n=== Sweep Summary ===")
    print(
        "Best validation accuracy "
        f"({best_eval_accuracy:.4f}) achieved with "
        f"lr={best_run_info['learning_rate']}, "
        f"wd={best_run_info['weight_decay']}, "
        f"temp={best_run_info['temperature']}"
    )
    print(f"Best temperature from sweep: {best_run_info['temperature']}")
else:
    print(
        "\n=== Sweep Summary ===\n"
        "No eval_accuracy found in eval metrics; best temperature unavailable."
    )