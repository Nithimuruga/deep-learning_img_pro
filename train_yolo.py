from pathlib import Path
import os

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "weapon_dataset_yolo" / "data.yaml"
OUTPUT_DIR = BASE_DIR / "models"


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def train() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}. Run prepare_dataset.py first."
        )

    fast_mode = env_bool("YOLO_FAST", False)
    continue_training = env_bool("YOLO_CONTINUE", False)

    model_name = os.getenv("YOLO_MODEL", "yolov8m.pt")
    continue_checkpoint = OUTPUT_DIR / "weapon_yolov8m" / "weights" / "last.pt"
    if continue_training and continue_checkpoint.exists():
        model_name = str(continue_checkpoint)

    epochs = env_int("YOLO_EPOCHS", 10 if fast_mode else 80)
    imgsz = env_int("YOLO_IMGSZ", 512 if fast_mode else 640)
    batch = env_int("YOLO_BATCH", 8 if fast_mode else 6)
    workers = env_int("YOLO_WORKERS", 2)
    patience = env_int("YOLO_PATIENCE", 6 if fast_mode else 20)
    device = env_str("YOLO_DEVICE", "0")

    model = YOLO(model_name)

    # Transfer learning starts from pretrained yolov8m weights.
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(OUTPUT_DIR),
        name="weapon_yolov8m",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        patience=patience,
        workers=workers,
        cache=True,
        cos_lr=True,
        close_mosaic=10,
        amp=True,
    )

    print("\nTraining completed.")
    print(f"Device used: {device}")
    print(f"Start checkpoint: {model_name}")
    print(f"Best model should be available in: {OUTPUT_DIR / 'weapon_yolov8m' / 'weights' / 'best.pt'}")


def validate() -> None:
    best_model = OUTPUT_DIR / "weapon_yolov8m" / "weights" / "best.pt"
    if not best_model.exists():
        print("Best model not found yet. Train first.")
        return

    model = YOLO(str(best_model))
    metrics = model.val(data=str(DATA_YAML))
    print("Validation finished.")
    print(metrics)


if __name__ == "__main__":
    train()
