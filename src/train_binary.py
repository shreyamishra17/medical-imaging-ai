import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from data.chexpert_binary import (
    ChexpertBinaryDataset,
    load_chexpert_binary_dataframe,
)
from models.resnet18_binary import build_resnet18_binary


def _select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _compute_pos_weight(labels: np.ndarray) -> torch.Tensor:
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


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
        labels = labels.float().to(device)
        logits = model(imgs).squeeze(1)
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
) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    y_true = []
    y_prob = []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        labels = labels.float().to(device)
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        losses.append(loss.item() * imgs.size(0))
        y_true.append(labels.cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    val_loss = float(np.sum(losses) / len(loader.dataset))

    auc = None
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    return val_loss, auc, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CheXpert binary baseline.")
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
    parser.add_argument("--out_dir", default="runs/binary_baseline")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = _select_device(args.device)

    exclude_uncertain = args.exclude_uncertain and not args.include_uncertain
    train_df, _ = load_chexpert_binary_dataframe(
        args.train_csv, args.data_root, exclude_uncertain=exclude_uncertain
    )
    val_df, _ = load_chexpert_binary_dataframe(
        args.val_csv, args.data_root, exclude_uncertain=exclude_uncertain
    )

    train_ds = ChexpertBinaryDataset(train_df)
    val_ds = ChexpertBinaryDataset(val_df)
    if len(train_ds) == 0:
        raise ValueError(
            "No training samples after filtering; check data_root and CSV paths."
        )
    if len(val_ds) == 0:
        raise ValueError(
            "No validation samples after filtering; check data_root and CSV paths."
        )

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

    model = build_resnet18_binary(pretrained=True).to(device)

    pos_weight = _compute_pos_weight(train_df["binary_label"].values)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metric = None
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_acc = _eval_epoch(model, val_loader, criterion, device)

        metric_str = f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        if val_auc is not None:
            metric_str += f" val_auc={val_auc:.4f}"
        print(metric_str)

        metric = val_auc if val_auc is not None else -val_loss
        if best_metric is None or metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                best_path,
            )

    print(f"Saved best checkpoint to {best_path}")


if __name__ == "__main__":
    main()
