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

    # Training settings
    # Smoke run (~30 min on RTX 4080): 1 epoch, no accum, faster eval/logging.
    # For full 10h run: set SWEEP_NUM_EPOCHS to 10–12, OUTPUT_DIR, GRADIENT_ACCUMULATION_STEPS=2
    # if you want effective batch 256, and point BEST_MODEL_PATH to the new checkpoint.
    OUTPUT_DIR = "./clip-finetuned/smoke-1epoch"
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 1
    TRAIN_BATCH_SIZE = 128
    EVAL_BATCH_SIZE = 256
    WEIGHT_DECAY = 0.1
    LOG_STEPS = 50
    NUM_WORKERS = 4
    GRADIENT_ACCUMULATION_STEPS = 1
    LR_SCHEDULER_TYPE = "cosine"
    WARMUP_RATIO = 0.05
    # Single combo (same targets as planned long run: lr 2.5e-4, wd 0.01, temp 0.05)
    LR_SWEEP = [2.5e-4]
    WEIGHT_DECAY_SWEEP = [0.01]
    TEMPERATURE_SWEEP = [0.05]
    SWEEP_NUM_EPOCHS = 1

    # Evaluation & Prediction settings
    # After smoke: set BEST_MODEL_PATH to .../smoke-1epoch/<run_name>/checkpoint-*/model.safetensors
    BEST_MODEL_PATH = (
        # "./clip-finetuned/2026-03-27_17-15-41/checkpoint-3125/model.safetensors"
        # "./clip-finetuned/lr_3e-04_2026-04-27_19-15-21/checkpoint-9375/model.safetensors"
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
    EVAL_PUSHABLE_DIR = "./eval-reports/smoke-1epoch"
    EVAL_PUSHABLE_TOP_N = 200
