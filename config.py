"""
Configuration file for CLIP model training and evaluation.
All hyperparameters and settings are defined here for easy modification.
"""


class Config:
    # Model settings
    IMAGE_ENCODER = "resnet"  # Options: "resnet", "vit"
    EMBED_DIM = 512  # Dimension of the joint embedding space
    # Temperature (match active TEMPERATURE_SWEEP value for eval/predict consistency)
    TEMPERATURE = 0.05
    # Fine-tuning controls: unfreeze top blocks/layers of encoders.
    UNFREEZE_IMAGE_BLOCKS = 1
    UNFREEZE_TEXT_LAYERS = 1

    # Data settings
    IMGNET_DIR = "./data/imagenet"
    VAL_SPLIT = 0.2
    # Templates for text augmentation during training
    TEXT_TEMPLATES = [
        "a photo of {}.",
        "a centered photo of {}.",
        "a close-up photo of {}.",
        "a cropped photo of {}.",
    ]

    # Training settings — full ~10h run (RTX 4080): cosine + warmup, effective batch 256 (128×2)
    OUTPUT_DIR = "./clip-finetuned/full-run_lr2.5e-04_wd0.01_epochs11"
    LEARNING_RATE = 2.5e-4
    NUM_EPOCHS = 11
    TRAIN_BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 256
    WEIGHT_DECAY = 0.01
    LOG_STEPS = 10
    NUM_WORKERS = 4
    GRADIENT_ACCUMULATION_STEPS = 2
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.05
    # Single run (no sweep grid)
    LR_SWEEP = [2.5e-4]
    WEIGHT_DECAY_SWEEP = [0.01]
    TEMPERATURE_SWEEP = [0.05]
    SWEEP_NUM_EPOCHS = 11

    # Evaluation & Prediction settings
    # After training: set to best checkpoint, e.g. .../full-run_.../checkpoint-*/model.safetensors
    BEST_MODEL_PATH = (
        "./clip-finetuned/lr_3e-04_wd_0.01_temp_0.1_2026-04-27_22-19-13/checkpoint-4689/model.safetensors"
    )
    TEST_DIR = "./data/testset"
    EVAL_TEXT_TEMPLATES = [
        "a photo of {}.",
        "a centered photo of {}.",
        "a close-up photo of {}.",
        "a cropped photo of {}.",
    ]
    PRED_PATH = "./data/prediction.json"
    EVAL_PUSHABLE_DIR = "./eval-reports/full-run"
    EVAL_PUSHABLE_TOP_N = 200
