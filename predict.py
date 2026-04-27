import json
import os

import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config
from data.dataset import get_eval_transform
from model.clip import CLIP


class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filenames = sorted(
            f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.root, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


# DO NOT change the mapping of class IDs to names
class_dict = {
    0: "Arctic fox, white fox, Alopex lagopus",
    1: "Australian terrier",
    2: "altar",
    3: "ballplayer, baseball player",
    4: "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    5: "bearskin, busby, shako",
    6: "bell pepper",
    7: "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    8: "bustard",
    9: "clog, geta, patten, sabot",
    10: "cocktail shaker",
    11: "confectionery, confectionary, candy store",
    12: "coral fungus",
    13: "corn",
    14: "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    15: "dam, dike, dyke",
    16: "desktop computer",
    17: "four-poster",
    18: "gas pump, gasoline pump, petrol pump, island dispenser",
    19: "goldfish, Carassius auratus",
    20: "hair spray",
    21: "harvester, reaper",
    22: "isopod",
    23: "lotion",
    24: "mashed potato",
    25: "meerkat, mierkat",
    26: "mud turtle",
    27: "necklace",
    28: "oboe, hautboy, hautbois",
    29: "orangutan, orang, orangutang, Pongo pygmaeus",
    30: "otter",
    31: "pencil sharpener",
    32: "plane, carpenter's plane, woodworking plane",
    33: "pop bottle, soda bottle",
    34: "puffer, pufferfish, blowfish, globefish",
    35: "quilt, comforter, comfort, puff",
    36: "screen, CRT screen",
    37: "silky terrier, Sydney silky",
    38: "sleeping bag",
    39: "snow leopard, ounce, Panthera uncia",
    40: "spatula",
    41: "spotted salamander, Ambystoma maculatum",
    42: "strawberry",
    43: "tarantula",
    44: "thresher, thrasher, threshing machine",
    45: "unicycle, monocycle",
    46: "warplane, military plane",
    47: "whiptail, whiptail lizard",
    48: "wood rabbit, cottontail, cottontail rabbit",
    49: "yurt",
}

# Text prompts for zero-shot classification
class_names = [class_dict[i] for i in range(len(class_dict))]
prompt_sets = [
    [template.format(name) for name in class_names]
    for template in Config.EVAL_TEXT_TEMPLATES
]

transform = get_eval_transform()
model = CLIP(
    encoder_type=Config.IMAGE_ENCODER,
    embed_dim=Config.EMBED_DIM,
    temperature=Config.TEMPERATURE,
    pretrained=False,
    trainable_image_blocks=Config.UNFREEZE_IMAGE_BLOCKS,
    trainable_text_layers=Config.UNFREEZE_TEXT_LAYERS,
)
state_dict = load_file(Config.BEST_MODEL_PATH)
model.load_state_dict(state_dict)
model = model.to("cuda")
model.eval()

# Dataset and DataLoader for flat testset
test_set = TestDataset(Config.TEST_DIR, transform=transform)
dataloader = DataLoader(
    test_set,
    batch_size=Config.EVAL_BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
)

# Run predictions
results = []
with torch.no_grad():
    for images, filenames in tqdm(dataloader, desc="Predicting"):
        images = images.to("cuda")
        logits_per_template = [
            model.compute_similarity(images, texts) for texts in prompt_sets
        ]
        ensemble_logits = torch.stack(logits_per_template, dim=0).mean(dim=0)
        probs = torch.softmax(ensemble_logits, dim=1)
        top10_ids = torch.topk(probs, k=10, dim=-1).indices.cpu().tolist()
        for fname, ids in zip(filenames, top10_ids):
            results.append(
                {
                    "filename": fname,
                    "top10_ids": ids,
                }
            )

# Save predictions
with open(Config.PRED_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} predictions to {Config.PRED_PATH}")
