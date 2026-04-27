"""
Configuration file for CLIP model training and evaluation.
All hyperparameters and settings are defined here for easy modification.
"""


class Config:
    # Model settings
    IMAGE_ENCODER = "resnet"  # Options: "resnet", "vit"
    EMBED_DIM = 512  # Dimension of the joint embedding space
    # Temperature parameter for contrastive loss
    TEMPERATURE = 0.07

    # Data settings
    IMGNET_DIR = "./data/imagenet"
    VAL_SPLIT = 0.2
    # Templates for text augmentation during training
    TEXT_TEMPLATES = ["a photo of {}."]

    # Training settings
    OUTPUT_DIR = "./clip-finetuned"
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 5
    TRAIN_BATCH_SIZE = 64
    EVAL_BATCH_SIZE = 64
    WEIGHT_DECAY = 0.1
    LOG_STEPS = 10
    NUM_WORKERS = 4
    # Sweep settings
    LR_SWEEP = [3e-5, 1e-4, 3e-4]
    SWEEP_NUM_EPOCHS = 5

    # Evaluation & Prediction settings
    BEST_MODEL_PATH = (
        "./clip-finetuned/2026-03-27_17-15-41/checkpoint-3125/model.safetensors"
    )
    TEST_DIR = "./data/testset"
    EVAL_TEXT_TEMPLATE = "a photo of {}."
    PRED_PATH = "./data/prediction.json"
    EVAL_PUSHABLE_DIR = "./eval-reports"
    EVAL_PUSHABLE_TOP_N = 200
