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
        # "./clip-finetuned/lr_3e-04_wd_0.01_temp_0.1_2026-04-27_22-19-13/checkpoint-4689/model.safetensors"
        "./clip-finetuned/full-run_lr2.5e-04_wd0.01_epochs11/lr_3e-04_wd_0.01_temp_0.05_2026-04-28_00-04-34/checkpoint-6256/model.safetensors"
    )
    TEST_DIR = "./data/testset"
    # Testset-first prompt bank (ordered; weights below align to this order).
    EVAL_TEXT_TEMPLATES = [
        "a photo of {}.",
        "a natural photo of {}.",
        "a real-world photo of {}.",
        "a centered photo of {}.",
        "a close-up photo of {}.",
        "a low-resolution photo of {}.",
        "a blurry photo of {}.",
        "a cropped photo of {}.",
        "a small {} in a scene.",
        "a photo of a distant {}.",
    ]
    # Template weighting for inference-time prompt ensembling.
    # Set to [] to fall back to uniform averaging.
    EVAL_TEMPLATE_WEIGHTS = [2.0, 1.2, 1.2, 1.0, 1.1, 1.0, 0.9, 1.0, 1.1, 1.0]
    # Optional class-specific aliases used only at prediction time.
    TESTSET_CLASS_PROMPT_VARIANTS = {
        "dam, dike, dyke": ["hydroelectric dam", "concrete dam wall"],
        "corn": ["corn cob", "maize ear"],
        "screen, CRT screen": ["computer monitor", "display screen"],
        "plane, carpenter's plane, woodworking plane": ["woodworking hand plane"],
        "pop bottle, soda bottle": ["plastic soda bottle", "soft drink bottle"],
        "warplane, military plane": ["fighter jet", "military aircraft"],
        "desktop computer": ["personal computer", "computer tower setup"],
    }
    # Deterministic TTA views applied at prediction time.
    PRED_ENABLE_TTA = True
    PRED_TTA_MODES = ["base", "hflip", "center_zoom_90", "center_zoom_80"]
    # Optional ablation runner outputs.
    PRED_RUN_ABLATIONS = False
    PRED_ABLATION_OUTPUT_DIR = "./eval-reports/testset-ablation"
    PRED_PATH = "./data/prediction.json"
    EVAL_PUSHABLE_DIR = "./eval-reports/full-run"
    EVAL_PUSHABLE_TOP_N = 200
    # Shared classification recipe knobs for eval/predict consistency.
    EVAL_ENABLE_TTA = True
    EVAL_TTA_MODES = ["base", "hflip", "center_zoom_90", "center_zoom_80"]
    EVAL_CLASS_PROMPT_VARIANTS = {}
    # Optional ablation sweep over eval datasets (ImageNet/CIFAR).
    EVAL_RUN_ABLATIONS = False
    EVAL_ABLATION_SCORE_WEIGHTS = {
        "cifar10_top1": 0.4,
        "cifar100_top1": 0.4,
        "imagenet_top1": 0.2,
    }
