"""
Evaluate a trained chest X-ray model on a chosen split (val or test).

Usage examples:
  # ResNet18 frozen
  python src/evaluate.py --split test --arch resnet18 --model_path /Users/shreyamishra/Projects/medical_imaging_ai/models/resnet18_chest_xray_frozen.pth

  # DenseNet121 frozen
  python src/evaluate.py --split test --arch densenet121 --model_path /Users/shreyamishra/Projects/medical_imaging_ai/models/densenet121_chest_xray_frozen.pth

What it does:
- Loads your saved model (.pth)
- Loads all images from the chosen folder (val/ or test/)
- Runs inference on the entire split
- Prints confusion matrix + classification report
- Saves results to reports/ as a text file (timestamped)
"""

import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights,
    densenet121, DenseNet121_Weights,
)
from sklearn.metrics import confusion_matrix, classification_report
from PIL import UnidentifiedImageError


# -------------------
# CONFIG (edit only if your paths change)
# -------------------
DATASET_ROOT = "/Users/shreyamishra/Documents/chest_xray_split"
BATCH_SIZE = 32

# Optional defaults (you can override via CLI)
DEFAULT_ARCH = "resnet18"
DEFAULT_MODEL_PATH = "/Users/shreyamishra/Projects/medical_imaging_ai/models/resnet18_chest_xray_frozen.pth"


# -------------------
# Device: Apple Silicon GPU (MPS) if available, else CPU
# -------------------
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


# -------------------
# Transforms (must match training/validation)
# -------------------
transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------
# Safe dataset loader (skips corrupted/unreadable images)
# -------------------
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            return self.__getitem__((index + 1) % len(self))


def build_model(arch: str) -> nn.Module:
    """Builds a 2-class model with the requested architecture."""
    arch = arch.lower()

    if arch == "resnet18":
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m

    if arch == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.DEFAULT)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        return m

    raise ValueError(f"Unknown arch: {arch}. Choose from: resnet18, densenet121")


def load_model(weights_path: str, arch: str) -> nn.Module:
    """Builds the requested architecture (2 classes) and loads saved weights."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}")

    model = build_model(arch)

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def evaluate_split(split_dir: str, arch: str, weights_path: str, batch_size: int = 32):
    """Runs evaluation on a folder like .../val or .../test."""
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    ds = SafeImageFolder(root=split_dir, transform=transform_eval)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = load_model(weights_path, arch)
    class_names = ds.classes

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    return {
        "device": str(device),
        "arch": arch,
        "model_path": weights_path,
        "num_images": len(ds),
        "class_names": class_names,
        "confusion_matrix": cm,
        "classification_report": report
    }


def save_report(split_name: str, results: dict):
    """Saves metrics to a timestamped report file."""
    os.makedirs("reports", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_arch = results["arch"].replace("/", "_")
    out_path = os.path.join("reports", f"evaluation_{safe_arch}_{split_name}_{ts}.txt")

    with open(out_path, "w") as f:
        f.write(f"Split: {split_name}\n")
        f.write(f"Device: {results['device']}\n")
        f.write(f"Arch: {results['arch']}\n")
        f.write(f"Model path: {results['model_path']}\n")
        f.write(f"Images: {results['num_images']}\n")
        f.write(f"Classes: {results['class_names']}\n\n")

        f.write("--- Confusion Matrix ---\n")
        f.write(np.array2string(results["confusion_matrix"]))
        f.write("\n\n--- Classification Report ---\n")
        f.write(results["classification_report"])
        f.write("\n")

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val",
                        help="Which split to evaluate: val or test")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--arch", choices=["resnet18", "densenet121"], default=DEFAULT_ARCH,
                        help="Model architecture")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to .pth model weights")
    args = parser.parse_args()

    split_dir = os.path.join(DATASET_ROOT, args.split)

    results = evaluate_split(
        split_dir=split_dir,
        arch=args.arch,
        weights_path=args.model_path,
        batch_size=args.batch_size
    )

    print(f"\nDevice: {results['device']}")
    print(f"Arch: {results['arch']}")
    print(f"Model path: {results['model_path']}")
    print(f"Split folder: {split_dir}")
    print(f"Images: {results['num_images']}")
    print(f"Classes: {results['class_names']}")

    print("\n--- Confusion Matrix ---")
    print(results["confusion_matrix"])

    print("\n--- Classification Report ---")
    print(results["classification_report"])

    report_path = save_report(args.split, results)
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
