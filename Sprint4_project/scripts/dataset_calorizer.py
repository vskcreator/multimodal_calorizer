import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ======================================
# DATASET
# ======================================

class CalorieDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ===== TEXT =====
        text = row["ingredients_text"]

        # ===== IMAGE =====
        dish_id = str(row["dish_id"])

        image_path = os.path.join(
            "data/images",
            dish_id,
            "rgb.png"
        )

        image = Image.open(image_path).convert("RGB")

        if self.transforms:
            image = np.array(image)
            image = self.transforms(image=image)["image"]

        # ===== MASS =====
        mass = torch.tensor(
            row["total_mass"],
            dtype=torch.float32
        )

        # ===== LABEL =====
        label = torch.tensor(
            row["total_calories"],
            dtype=torch.float32
        )

        return {
            "text": text,
            "image": image,
            "mass": mass,
            "label": label
        }


# ======================================
# COLLATE FUNCTION
# ======================================

def collate_fn(batch, tokenizer):

    texts = [x["text"] for x in batch]
    images = torch.stack([x["image"] for x in batch])
    labels = torch.stack([x["label"] for x in batch])
    masses = torch.stack([x["mass"] for x in batch])

    tokenized = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "image": images,
        "mass": masses,
        "label": labels
    }


# ======================================
# TRANSFORMS (В КОНЦЕ)
# ======================================

def get_transforms(config, ds_type="train"):

    SIZE = 224

    if ds_type == "train":
        return A.Compose([
            A.SmallestMaxSize(max_size=SIZE),
            A.CenterCrop(height=SIZE, width=SIZE),

            # можно включить позже:
            # A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.3),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=SIZE),
            A.CenterCrop(height=SIZE, width=SIZE),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
