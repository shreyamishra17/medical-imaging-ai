import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix
from PIL import UnidentifiedImageError

DATASET_ROOT = "/Users/shreyamishra/Documents/chest_xray"
#MODEL_PATH = "/Users/shreyamishra/Projects/medical_imaging_ai/models/resnet18_chest_xray.pth"
MODEL_PATH = "/Users/shreyamishra/Projects/medical_imaging_ai/models/resnet18_chest_xray_frozen.pth"
BATCH_SIZE = 32

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            return self.__getitem__((index + 1) % len(self))

def load_model():
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    state = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    return m

def sweep_thresholds(split="test"):
    split_dir = os.path.join(DATASET_ROOT, split)
    ds = SafeImageFolder(split_dir, transform=transform_eval)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # class order from ImageFolder is alphabetical, usually: ['NORMAL','PNEUMONIA']
    class_names = ds.classes
    pneumonia_idx = class_names.index("PNEUMONIA")
    normal_idx = class_names.index("NORMAL")

    model = load_model()

    probs_pneu = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_pneu.extend(probs[:, pneumonia_idx])
            labels.extend(y.numpy())

    probs_pneu = np.array(probs_pneu)
    labels = np.array(labels)

    # We'll treat "pneumonia" as positive class
    # Predict pneumonia if prob >= threshold else normal
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    print(f"\nSplit: {split}  Images: {len(ds)}  Device: {device}")
    print("Threshold | Acc   | Normal Recall | Pneu Recall | Pneu Precision | FP(normal->pneu) | FN(pneu->normal)")
    print("-"*105)

    for t in thresholds:
        preds = np.where(probs_pneu >= t, pneumonia_idx, normal_idx)

        # Confusion matrix arranged by labels [NORMAL, PNEUMONIA]
        cm = confusion_matrix(labels, preds, labels=[normal_idx, pneumonia_idx])
        tn, fp, fn, tp = cm.ravel()

        acc = (tp + tn) / (tp + tn + fp + fn)

        # Normal recall = TN / (TN + FP)  (how many normals correctly called normal)
        normal_recall = tn / (tn + fp) if (tn + fp) else 0.0

        # Pneumonia recall = TP / (TP + FN)
        pneu_recall = tp / (tp + fn) if (tp + fn) else 0.0

        # Pneumonia precision = TP / (TP + FP)
        pneu_precision = tp / (tp + fp) if (tp + fp) else 0.0

        print(f"{t:8.2f} | {acc:5.3f} | {normal_recall:12.3f} | {pneu_recall:10.3f} | {pneu_precision:13.3f} | {fp:14d} | {fn:13d}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()
    sweep_thresholds(args.split)

if __name__ == "__main__":
    main()
