import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score

from data.kermany_pneumonia import KermanyPneumoniaDataset
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
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
    return val_loss, auc, acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kermany pneumonia baseline.")
    parser.add_argument("--data_root", default="datasets/chest_xray")
    parser.add_argument("--train_dir", default="train")
    parser.add_argument("--val_dir", default="val")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out_dir", default="runs/pneumonia_baseline")
    parser.add_argument("--lungmask_model", default=None)
    parser.add_argument("--lungmask_cache_dir", default=None)
    parser.add_argument("--lungmask_device", default="cpu")
    args = parser.parse_args()

    device = _select_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.data_root, args.train_dir)
    val_path = os.path.join(args.data_root, args.val_dir)

    train_ds = KermanyPneumoniaDataset(
        train_path,
        lungmask_model=args.lungmask_model,
        lungmask_cache_dir=args.lungmask_cache_dir,
        device=args.lungmask_device,
    )
    val_ds = KermanyPneumoniaDataset(
        val_path,
        lungmask_model=args.lungmask_model,
        lungmask_cache_dir=args.lungmask_cache_dir,
        device=args.lungmask_device,
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train/val dataset is empty; check dataset paths.")

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

    labels = np.array([label for _, label in train_ds.ds.samples], dtype=np.int64)
    pos_weight = _compute_pos_weight(labels)

    model = build_resnet18_binary(pretrained=True).to(device)
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
