import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from data.chexpert_binary import ChexpertBinaryDataset, load_chexpert_binary_dataframe
from models.resnet18_binary import build_resnet18_binary, load_checkpoint


def _select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _predict(model, loader, device):
    model.eval()
    y_true = []
    y_prob = []
    paths = []
    for imgs, labels, img_paths in loader:
        imgs = imgs.to(device)
        logits = model(imgs).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.append(probs)
        y_true.append(labels.numpy())
        paths.extend(img_paths)
    return (
        np.concatenate(y_true),
        np.concatenate(y_prob),
        paths,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CheXpert binary model.")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--csv", default="datasets/valid.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--exclude_uncertain", action="store_true", default=True)
    parser.add_argument("--include_uncertain", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    device = _select_device(args.device)

    exclude_uncertain = args.exclude_uncertain and not args.include_uncertain
    df, _ = load_chexpert_binary_dataframe(
        args.csv, args.data_root, exclude_uncertain=exclude_uncertain
    )
    ds = ChexpertBinaryDataset(df)
    if len(ds) == 0:
        raise ValueError(
            "No evaluation samples after filtering; check data_root and CSV paths."
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type in ["cuda", "mps"]),
    )

    model = build_resnet18_binary(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)

    y_true, y_prob, paths = _predict(model, loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")
    print(f"Precision: {prec:.4f} Recall: {rec:.4f} F1: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")

    out_csv = args.out_csv
    if out_csv is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        out_csv = os.path.join(os.path.dirname(args.checkpoint), f"{base}_report.csv")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "y_true", "y_pred_prob", "y_pred_label"])
        for p, yt, yp, yl in zip(paths, y_true, y_prob, y_pred):
            writer.writerow([p, int(yt), float(yp), int(yl)])

    print(f"Saved report to {out_csv}")


if __name__ == "__main__":
    main()
