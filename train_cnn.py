from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "weapon_detection"
OUTPUT_DIR = BASE_DIR / "models" / "cnn_weapon_classifier"
CLASS_NAMES = ["gun", "knife"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def resolve_device() -> torch.device:
    requested = os.getenv("CNN_DEVICE", "").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Sample:
    image_path: Path
    class_id: int


class WeaponDataset(Dataset):
    def __init__(self, samples: List[Sample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as img:
            image = img.convert("RGB")
        return self.transform(image), sample.class_id


class WeaponCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def gather_samples(split: str) -> List[Sample]:
    samples: List[Sample] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        split_dir = DATASET_DIR / class_name / split
        if not split_dir.exists():
            continue
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VALID_EXT:
                continue
            samples.append(Sample(image_path=path, class_id=class_id))
    return samples


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
) -> Tuple[float, float]:
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    avg_loss = total_loss / max(1, total_seen)
    accuracy = total_correct / max(1, total_seen)
    return avg_loss, accuracy


def main() -> None:
    epochs = env_int("CNN_EPOCHS", 12)
    batch_size = env_int("CNN_BATCH", 32)
    img_size = env_int("CNN_IMGSZ", 224)
    num_workers = env_int("CNN_WORKERS", 2)
    learning_rate = env_float("CNN_LR", 1e-3)
    weight_decay = env_float("CNN_WEIGHT_DECAY", 1e-4)
    device = resolve_device()

    train_samples = gather_samples("train")
    val_samples = gather_samples("val")

    if not train_samples:
        raise RuntimeError(
            f"No training images found in {DATASET_DIR}. Expected folders like weapon_detection/gun/train and weapon_detection/knife/train"
        )
    if not val_samples:
        raise RuntimeError(
            f"No validation images found in {DATASET_DIR}. Expected folders like weapon_detection/gun/val and weapon_detection/knife/val"
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = WeaponDataset(train_samples, train_transform)
    val_ds = WeaponDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_csv = OUTPUT_DIR / "metrics.csv"
    best_path = OUTPUT_DIR / "best.pt"
    last_path = OUTPUT_DIR / "last.pt"
    meta_path = OUTPUT_DIR / "meta.json"

    model = WeaponCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_val_acc = -1.0

    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
            with torch.no_grad():
                val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_acc:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}"])
            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": CLASS_NAMES,
                "img_size": img_size,
                "val_acc": float(val_acc),
            }
            torch.save(checkpoint, last_path)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(checkpoint, best_path)

    meta = {
        "dataset_dir": str(DATASET_DIR),
        "class_names": CLASS_NAMES,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "device": str(device),
        "best_val_acc": round(float(best_val_acc), 6),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nCNN training completed.")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Metrics: {metrics_csv}")


if __name__ == "__main__":
    main()
