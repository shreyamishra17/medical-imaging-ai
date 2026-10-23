import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from data.chexpert_multilabel import (
    ChexpertMultiLabelDataset,
    load_chexpert_multilabel_dataframe,
)
from models.resnet18_multilabel import build_resnet18_multilabel


def _select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    weight = np.ones_like(pos, dtype=np.float32)
    nonzero = pos > 0
    weight[nonzero] = (neg[nonzero] / pos[nonzero]).astype(np.float32)
    return torch.tensor(weight, dtype=torch.float32)


def _load_backbone_weights(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    filtered = {}
    for k, v in state.items():
        if k.startswith("fc."):
            continue
        filtered[k] = v
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"Missing keys when loading backbone: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading backbone: {unexpected}")


def _compute_auc(y_true: np.ndarray, y_prob: np.ndarray, label_names) -> Tuple[Dict[str, float], float, float]:
    label_auc = {}
    for idx, name in enumerate(label_names):
        if len(np.unique(y_true[:, idx])) < 2:
            continue
        label_auc[name] = float(roc_auc_score(y_true[:, idx], y_prob[:, idx]))
    macro = float(np.mean(list(label_auc.values()))) if label_auc else None
    micro = None
    if len(np.unique(y_true)) > 1:
        try:
            micro = float(roc_auc_score(y_true, y_prob, average="micro"))
        except ValueError:
            micro = None
    return label_auc, macro, micro


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names,
) -> Tuple[float, Dict[str, float], float, float]:
    model.eval()
    losses = []
    y_true = []
    y_prob = []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        losses.append(loss.item() * imgs.size(0))
        y_true.append(labels.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    val_loss = float(np.sum(losses) / len(loader.dataset))
    label_auc, macro_auc, micro_auc = _compute_auc(y_true, y_prob, label_names)
    return val_loss, label_auc, macro_auc, micro_auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CheXpert multi-label baseline.")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--train_csv", default="datasets/train.csv")
    parser.add_argument("--val_csv", default="datasets/valid.csv")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--exclude_uncertain", action="store_true", default=True)
    parser.add_argument("--include_uncertain", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lungmask_model", default=None)
    parser.add_argument("--lungmask_cache_dir", default="runs/lung_masks")
    parser.add_argument("--lungmask_device", default="cpu")
    parser.add_argument("--init_checkpoint", default=None)
    parser.add_argument("--out_dir", default="runs/multilabel_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = _select_device(args.device)

    exclude_uncertain = args.exclude_uncertain and not args.include_uncertain
    train_df, label_names = load_chexpert_multilabel_dataframe(
        args.train_csv, args.data_root, exclude_uncertain=exclude_uncertain
    )
    val_df, _ = load_chexpert_multilabel_dataframe(
        args.val_csv, args.data_root, exclude_uncertain=exclude_uncertain
    )

    train_ds = ChexpertMultiLabelDataset(
        train_df,
        lungmask_model=args.lungmask_model,
        lungmask_cache_dir=args.lungmask_cache_dir,
        device=args.lungmask_device,
    )
    val_ds = ChexpertMultiLabelDataset(
        val_df,
        lungmask_model=args.lungmask_model,
        lungmask_cache_dir=args.lungmask_cache_dir,
        device=args.lungmask_device,
    )
    if len(train_ds) == 0:
        raise ValueError("No training samples after filtering; check data_root and CSV paths.")
    if len(val_ds) == 0:
        raise ValueError("No validation samples after filtering; check data_root and CSV paths.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type in ["cuda", "mps"]),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type in ["cuda", "mps"]),
    )

    model = build_resnet18_multilabel(num_labels=len(label_names), pretrained=True).to(device)
    if args.init_checkpoint:
        _load_backbone_weights(model, args.init_checkpoint)

    pos_weight = _compute_pos_weight(np.vstack(train_df["labels"].values))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metric = None
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, label_auc, macro_auc, micro_auc = _eval_epoch(
            model, val_loader, criterion, device, label_names
        )

        metric_str = f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        if macro_auc is not None:
            metric_str += f" macro_auc={macro_auc:.4f}"
        if micro_auc is not None:
            metric_str += f" micro_auc={micro_auc:.4f}"
        print(metric_str)

        metric = macro_auc if macro_auc is not None else -val_loss
        if best_metric is None or metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "label_auc": label_auc,
                    "label_names": label_names,
                    "args": vars(args),
                },
                best_path,
            )

    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
