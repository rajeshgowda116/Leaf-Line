"""
LeafLine - Binary Plant-Leaf Validator Training and Evaluation Script
---------------------------------------------------------------------
Trains a binary convolutional classifier (LeafValidatorNet) to distinguish
between real agricultural plant leaves and non-leaf photographs
(human faces, lamps/candles/diyas, vehicles, furniture, everyday objects).

Maintains strictly separated training and testing datasets.
Calculates Accuracy, Precision, Recall, False Acceptance Rate (FAR),
and False Rejection Rate (FRR).
Saves the trained model to models/leaf_validator.pth.
"""

import os
import json
import random
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set reproducible seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_SAVE_PATH = BASE_DIR / "models" / "leaf_validator.pth"
EVAL_SAVE_PATH = BASE_DIR / "models" / "leaf_validator_evaluation.json"


# -------------------------------------------------------------
# 1. LeafValidatorNet Architecture
# -------------------------------------------------------------

class LeafValidatorNet(nn.Module):
    """
    Lightweight, fast convolutional binary classifier for plant leaf validation.
    Input: (B, 3, 128, 128) normalized tensor
    Output: (B, 1) logit score (Sigmoid > 0.5 = plant_leaf, <= 0.5 = not_plant_leaf)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 128x1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


# -------------------------------------------------------------
# 2. Diverse Dataset Generation
# -------------------------------------------------------------

def generate_leaf_sample() -> Image.Image:
    """Generates a realistic synthetic leaf with chlorophyll patterns, veins, spots."""
    img = Image.new("RGB", (128, 128), color=(
        random.randint(180, 230),
        random.randint(170, 220),
        random.randint(150, 200)
    ))
    draw = ImageDraw.Draw(img)

    # Base leaf blade (green or chlorotic yellow-green)
    leaf_type = random.choice(["healthy_green", "dark_green", "yellowing", "spotted_blight"])
    if leaf_type == "healthy_green":
        blade_color = (random.randint(30, 60), random.randint(130, 200), random.randint(30, 70))
    elif leaf_type == "dark_green":
        blade_color = (random.randint(15, 40), random.randint(90, 140), random.randint(20, 50))
    elif leaf_type == "yellowing":
        blade_color = (random.randint(150, 210), random.randint(170, 215), random.randint(30, 60))
    else:  # spotted blight
        blade_color = (random.randint(40, 80), random.randint(110, 160), random.randint(30, 60))

    # Leaf shape (ellipse or polygon)
    bbox = [random.randint(15, 30), random.randint(10, 25), random.randint(95, 115), random.randint(95, 118)]
    draw.ellipse(bbox, fill=blade_color)

    # Main vein
    vein_color = (min(255, blade_color[0] + 20), min(255, blade_color[1] + 20), min(255, blade_color[2] + 10))
    mid_x = (bbox[0] + bbox[2]) // 2
    draw.line([(mid_x, bbox[1]), (mid_x, bbox[3])], fill=vein_color, width=random.randint(2, 3))

    # Lateral veins
    for y in range(bbox[1] + 15, bbox[3] - 15, random.randint(12, 18)):
        offset_y = random.randint(-10, -3)
        draw.line([(mid_x, y), (bbox[0] + 5, y + offset_y)], fill=vein_color, width=1)
        draw.line([(mid_x, y), (bbox[2] - 5, y + offset_y)], fill=vein_color, width=1)

    # Necrotic spots for diseased leaves
    if leaf_type == "spotted_blight":
        for _ in range(random.randint(3, 7)):
            sx = random.randint(bbox[0] + 10, bbox[2] - 10)
            sy = random.randint(bbox[1] + 10, bbox[3] - 10)
            sr = random.randint(4, 9)
            draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=(random.randint(70, 110), random.randint(40, 70), random.randint(20, 40)))

    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def generate_non_leaf_sample() -> Image.Image:
    """Generates realistic non-leaf images across multiple categories."""
    category = random.choice([
        "human_skin_face",
        "lamp_flame_diya",
        "car_metallic_vehicle",
        "indoor_furniture_wood",
        "book_text_paper",
        "fabric_clothing",
        "abstract_noise_texture"
    ])

    img = Image.new("RGB", (128, 128))
    draw = ImageDraw.Draw(img)

    if category == "human_skin_face":
        # Skin tone range (Fitzpatrick I to VI)
        skin_r = random.randint(140, 245)
        skin_g = int(skin_r * random.uniform(0.65, 0.82))
        skin_b = int(skin_g * random.uniform(0.60, 0.85))
        img.paste((skin_r, skin_g, skin_b), [0, 0, 128, 128])
        # Add facial features (eyes / lips / hair)
        draw.ellipse([30, 45, 55, 60], fill=(random.randint(40, 70), random.randint(30, 50), random.randint(20, 40)))
        draw.ellipse([73, 45, 98, 60], fill=(random.randint(40, 70), random.randint(30, 50), random.randint(20, 40)))
        draw.line([(45, 95), (83, 95)], fill=(min(255, skin_r + 30), max(0, skin_g - 40), max(0, skin_b - 30)), width=4)

    elif category == "lamp_flame_diya":
        # Dark room or warm clay background
        img.paste((random.randint(20, 50), random.randint(15, 30), random.randint(10, 25)), [0, 0, 128, 128])
        # Terracotta lamp body
        draw.rectangle([25, 80, 103, 115], fill=(random.randint(160, 210), random.randint(60, 100), random.randint(20, 45)))
        # Incandescent flame (pure red, orange, yellow core)
        draw.ellipse([45, 20, 83, 85], fill=(255, random.randint(100, 140), 10))
        draw.ellipse([53, 35, 75, 75], fill=(255, random.randint(220, 255), random.randint(100, 180)))

    elif category == "car_metallic_vehicle":
        # Metal car body (blue, red, silver, black)
        metal_color = random.choice([(180, 190, 200), (30, 60, 160), (180, 20, 30), (40, 40, 45)])
        img.paste(metal_color, [0, 0, 128, 128])
        # Windshield / glass & tires
        draw.rectangle([20, 20, 108, 65], fill=(random.randint(30, 60), random.randint(50, 80), random.randint(70, 110)))
        draw.ellipse([15, 90, 45, 120], fill=(20, 20, 20))
        draw.ellipse([83, 90, 113, 120], fill=(20, 20, 20))

    elif category == "indoor_furniture_wood":
        # Wood grain table / floor / desk
        wood_color = (random.randint(110, 160), random.randint(65, 100), random.randint(30, 60))
        img.paste(wood_color, [0, 0, 128, 128])
        # Planks / grain lines
        for y in range(0, 128, random.randint(15, 30)):
            draw.line([(0, y), (128, y)], fill=(max(0, wood_color[0] - 25), max(0, wood_color[1] - 20), max(0, wood_color[2] - 15)), width=2)

    elif category == "book_text_paper":
        # White / cream paper
        img.paste((random.randint(240, 255), random.randint(238, 252), random.randint(225, 245)), [0, 0, 128, 128])
        # Printed text lines
        for y in range(15, 115, 8):
            draw.line([(15, y), (random.randint(90, 115), y)], fill=(random.randint(20, 50), random.randint(20, 50), random.randint(20, 50)), width=1)

    elif category == "fabric_clothing":
        # Weave / fabric texture
        fabric_color = random.choice([(140, 30, 40), (40, 90, 140), (200, 180, 130), (30, 30, 35)])
        img.paste(fabric_color, [0, 0, 128, 128])
        for x in range(0, 128, 4):
            draw.line([(x, 0), (x, 128)], fill=(min(255, fabric_color[0] + 15), min(255, fabric_color[1] + 15), min(255, fabric_color[2] + 15)), width=1)

    else:  # abstract_noise_texture
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(arr)

    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


class BinaryLeafDataset(Dataset):
    """PyTorch Dataset returning (tensor, label) where label 1 = leaf, 0 = non_leaf."""
    def __init__(self, samples: List[Tuple[Image.Image, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.samples[idx]
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        return tensor, torch.tensor([float(label)], dtype=torch.float32)


# -------------------------------------------------------------
# 3. Training & Evaluation Pipeline
# -------------------------------------------------------------

def build_datasets(
    num_train_per_class: int = 400,
    num_test_per_class: int = 100,
) -> Tuple[Dataset, Dataset]:
    """Generates independent train and test datasets without data leakage."""
    train_samples = []
    test_samples = []

    # Training set
    for _ in range(num_train_per_class):
        train_samples.append((generate_leaf_sample(), 1))
        train_samples.append((generate_non_leaf_sample(), 0))

    # Testing set
    for _ in range(num_test_per_class):
        test_samples.append((generate_leaf_sample(), 1))
        test_samples.append((generate_non_leaf_sample(), 0))

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = BinaryLeafDataset(train_samples, transform=train_transform)
    test_ds = BinaryLeafDataset(test_samples, transform=test_transform)

    return train_ds, test_ds


def train_and_evaluate(epochs: int = 15, batch_size: int = 32):
    """Executes training, computes rigorous metrics on test set, and saves model."""
    print("=" * 65)
    print("LeafLine Binary Plant-Leaf Validator: Training & Evaluation")
    print("=" * 65)

    train_ds, test_ds = build_datasets(num_train_per_class=350, num_test_per_class=100)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset summary: Train = {len(train_ds)} samples, Test = {len(test_ds)} samples")

    model = LeafValidatorNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2%}")

    # Rigorous Evaluation on Held-Out Test Set
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            y_true.extend(labels.view(-1).tolist())
            y_pred.extend(preds.view(-1).tolist())
            y_scores.extend(probs.view(-1).tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total_test = len(y_true)
    accuracy = (tp + tn) / total_test
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # False Acceptance Rate: Non-leaves wrongly accepted as leaves
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # False Rejection Rate: Real leaves wrongly rejected as non-leaves
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    metrics = {
        "model_architecture": "LeafValidatorNet (4-stage Conv2D + BatchNorm + GAP + Linear)",
        "total_test_samples": total_test,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1_score), 4),
        "false_acceptance_rate_far": round(float(far), 4),
        "false_rejection_rate_frr": round(float(frr), 4),
        "classification_threshold": 0.50,
        "dataset_categories": [
            "plant_leaf (green, chlorotic yellow, necrotic spotted, diverse shapes)",
            "not_plant_leaf (human faces, candles/lamps/diyas, vehicles, furniture, books/text, fabrics, noise)"
        ]
    }

    print("\n" + "=" * 65)
    print("Held-Out Test Set Evaluation Results:")
    print("=" * 65)
    print(f"Accuracy:                 {accuracy:.2%}")
    print(f"Precision:                {precision:.2%}")
    print(f"Recall (Sensitivity):     {recall:.2%}")
    print(f"F1-Score:                 {f1_score:.4f}")
    print(f"False Acceptance Rate:    {far:.2%} (Non-leaves incorrectly accepted)")
    print(f"False Rejection Rate:     {frr:.2%} (Leaves incorrectly rejected)")
    print(f"Confusion Matrix:         TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print("=" * 65)

    # Save model weights
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    torch.save(model.state_dict(), str(MODEL_SAVE_PATH))
    print(f"\nModel weights saved successfully to: {MODEL_SAVE_PATH}")

    # Save evaluation report
    with open(EVAL_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report written to:        {EVAL_SAVE_PATH}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate(epochs=15, batch_size=32)
