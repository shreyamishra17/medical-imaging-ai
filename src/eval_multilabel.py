import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

from data.chexpert_multilabel import ChexpertMultiLabelDataset, load_chexpert_multilabel_dataframe
from models.resnet18_multilabel import build_resnet18_multilabel, load_checkpoint


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
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.append(probs)
        y_true.append(labels.numpy())
        paths.extend(img_paths)
    return np.concatenate(y_true), np.concatenate(y_prob), paths


def _compute_auc(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    label_auc = {}
    for idx, name in enumerate(label_names):
        if len(np.unique(y_true[:, idx])) < 2:
            continue
        label_auc[name] = float(roc_auc_score(y_true[:, idx], y_prob[:, idx]))
    return label_auc


def _best_f1_thresholds(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    thresholds = {}
    grid = np.linspace(0.05, 0.95, 19)
    for idx, name in enumerate(label_names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        if len(np.unique(yt)) < 2:
            thresholds[name] = 0.5
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            pred = (yp >= t).astype(int)
            tp = np.sum((pred == 1) & (yt == 1))
            fp = np.sum((pred == 1) & (yt == 0))
            fn = np.sum((pred == 0) & (yt == 1))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = (2 * precision * recall) / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[name] = best_t
    return thresholds


def _save_report_csv(
    out_csv: str,
    paths: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: List[str],
    thresholds: Dict[str, float],
) -> None:
    header = ["image_path"]
    for name in label_names:
        header.append(f"{name}_true")
        header.append(f"{name}_prob")
        header.append(f"{name}_pred")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, path in enumerate(paths):
            row = [path]
            for j, name in enumerate(label_names):
                prob = float(y_prob[i, j])
                pred = int(prob >= thresholds.get(name, 0.5))
                row.extend([int(y_true[i, j]), prob, pred])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CheXpert multi-label model.")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--csv", default="datasets/valid.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--exclude_uncertain", action="store_true", default=True)
    parser.add_argument("--include_uncertain", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lungmask_model", default=None)
    parser.add_argument("--lungmask_cache_dir", default="runs/lung_masks")
    parser.add_argument("--lungmask_device", default="cpu")
    parser.add_argument("--save_thresholds", action="store_true", default=True)
    parser.add_argument("--report_csv", default=None)
    args = parser.parse_args()

    device = _select_device(args.device)
    exclude_uncertain = args.exclude_uncertain and not args.include_uncertain

    df, label_names = load_chexpert_multilabel_dataframe(
        args.csv, args.data_root, exclude_uncertain=exclude_uncertain
    )
    ds = ChexpertMultiLabelDataset(
        df,
        lungmask_model=args.lungmask_model,
        lungmask_cache_dir=args.lungmask_cache_dir,
        device=args.lungmask_device,
    )
    if len(ds) == 0:
        raise ValueError("No evaluation samples after filtering; check data_root and CSV paths.")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type in ["cuda", "mps"]),
    )

    model = build_resnet18_multilabel(num_labels=len(label_names), pretrained=False).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    label_names = checkpoint.get("label_names", label_names)

    y_true, y_prob, paths = _predict(model, loader, device)

    label_auc = _compute_auc(y_true, y_prob, label_names)
    macro_auc = float(np.mean(list(label_auc.values()))) if label_auc else None
    micro_auc = None
    if len(np.unique(y_true)) > 1:
        try:
            micro_auc = float(roc_auc_score(y_true, y_prob, average="micro"))
        except ValueError:
            micro_auc = None

    print("AUC per label:")
    for name in label_names:
        if name in label_auc:
            print(f"  {name}: {label_auc[name]:.4f}")
    if macro_auc is not None:
        print(f"Macro AUC: {macro_auc:.4f}")
    if micro_auc is not None:
        print(f"Micro AUC: {micro_auc:.4f}")

    thresholds = {name: 0.5 for name in label_names}
    if args.save_thresholds:
        thresholds = _best_f1_thresholds(y_true, y_prob, label_names)
        out_path = os.path.join(os.path.dirname(args.checkpoint), "thresholds.json")
        with open(out_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"Saved thresholds to {out_path}")

    if args.report_csv is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        args.report_csv = os.path.join(os.path.dirname(args.checkpoint), f"{base}_multilabel_report.csv")

    _save_report_csv(args.report_csv, paths, y_true, y_prob, label_names, thresholds)
    print(f"Saved report to {args.report_csv}")


if __name__ == "__main__":
    main()
