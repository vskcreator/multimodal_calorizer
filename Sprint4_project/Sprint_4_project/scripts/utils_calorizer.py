import os
import random
import json
from functools import partial

import numpy as np
import pandas as pd

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics
from transformers import AutoModel, AutoTokenizer

from dataset_calorizer import (
    CalorieDataset,
    collate_fn,
    get_transforms
)


# ======================================
# SEED
# ======================================

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# ======================================
# MODEL
# ======================================

class MultimodalModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.text_model = AutoModel.from_pretrained(
            config.TEXT_MODEL_NAME
        )

        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )

        self.text_proj = nn.Linear(
            self.text_model.config.hidden_size,
            config.HIDDEN_DIM
        )

        self.image_proj = nn.Linear(
            self.image_model.num_features,
            config.HIDDEN_DIM
        )

        self.mass_proj = nn.Linear(1, config.HIDDEN_DIM)

        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM * 3,
                      config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1)
        )

    def forward(self, input_ids, attention_mask, image, mass):

        text_features = self.text_model(
            input_ids,
            attention_mask
        ).last_hidden_state[:, 0, :]

        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        mass = mass.unsqueeze(-1)
        mass_emb = self.mass_proj(mass)

        fused = torch.cat(
            [text_emb, image_emb, mass_emb],
            dim=1
        )

        output = self.regressor(fused)

        return output


# ======================================
# VALIDATION
# ======================================

def validate(model, loader, device, metric):

    model.eval()

    with torch.no_grad():

        for batch in loader:

            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device)
            }

            labels = batch["label"].to(device).float()

            logits = model(**inputs)
            preds = logits.squeeze(-1)

            _ = metric(preds, labels)

    return metric.compute().cpu().numpy()


# ======================================
# TRAIN
# ======================================

def train(config, device):

    seed_everything(config.SEED)

    model = MultimodalModel(config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.TEXT_MODEL_NAME
    )

    optimizer = AdamW([
        {"params": model.text_model.parameters(),
         "lr": config.TEXT_LR},

        {"params": model.image_model.parameters(),
         "lr": config.IMAGE_LR},

        {"params": model.regressor.parameters(),
         "lr": config.REGRESSOR_LR}
    ])

    criterion = nn.L1Loss()

    mae_train = torchmetrics.MeanAbsoluteError().to(device)
    mae_val = torchmetrics.MeanAbsoluteError().to(device)

    train_df = pd.read_csv(config.TRAIN_DF_PATH)
    val_df = pd.read_csv(config.VAL_DF_PATH)

    train_ds = CalorieDataset(
        train_df,
        get_transforms(config)
    )

    val_ds = CalorieDataset(
        val_df,
        get_transforms(config, ds_type="val")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    # ===============================
    # METRICS HISTORY
    # ===============================

    train_mae_history = []
    val_mae_history = []
    loss_history = []

    best_mae = 1e9

    print("training started")

    for epoch in range(config.EPOCHS):

        model.train()
        total_loss = 0

        for batch in train_loader:

            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device)
            }

            labels = batch["label"].to(device).float()

            optimizer.zero_grad()

            logits = model(**inputs)
            preds = logits.squeeze(-1)

            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _ = mae_train(preds, labels)

        train_mae = mae_train.compute().cpu().numpy()
        val_mae = validate(model, val_loader, device, mae_val)

        avg_loss = total_loss / len(train_loader)

        train_mae_history.append(float(train_mae))
        val_mae_history.append(float(val_mae))
        loss_history.append(float(avg_loss))

        mae_train.reset()
        mae_val.reset()

        print(
            f"Epoch {epoch} | "
            f"Loss {avg_loss:.3f} | "
            f"Train MAE {train_mae:.2f} | "
            f"Val MAE {val_mae:.2f}"
        )

        # SAVE BEST MODEL
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), config.SAVE_PATH)
            print(f"Best model saved → MAE {best_mae:.2f}")

        # EARLY STOP
        if val_mae <= config.TARGET_MAE:
            print(f"Target MAE reached ({val_mae:.2f}) → STOP")
            break

    # ===============================
    # SAVE METRICS TO FILE
    # ===============================

    history = {
        "train_mae": train_mae_history,
        "val_mae": val_mae_history,
        "loss": loss_history
    }

    with open("training_history.json", "w") as f:
        json.dump(history, f)

    print("Metrics saved → training_history.json")

    return history
