import os
from typing import List, Optional, Tuple

import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.lung_mask import LungMasker, cached_lung_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True

CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def _find_path_column(df: pd.DataFrame) -> str:
    for col in ["Path", "path", "image_path", "Image", "ImagePath"]:
        if col in df.columns:
            return col
    raise ValueError("Could not find a path column (expected 'Path').")


def _infer_label_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c in CHEXPERT_LABELS]
    if not cols:
        raise ValueError("No CheXpert label columns found in CSV.")
    if "No Finding" not in cols:
        raise ValueError("Missing required label column: 'No Finding'.")
    return cols


def _filter_frontal(df: pd.DataFrame, path_col: str) -> pd.DataFrame:
    if "Frontal/Lateral" in df.columns:
        mask = df["Frontal/Lateral"].astype(str).str.lower() == "frontal"
        return df[mask].copy()
    if path_col in df.columns:
        mask = df[path_col].astype(str).str.lower().str.contains("frontal")
        return df[mask].copy()
    return df


def _resolve_path(path_str: str, data_root: str, csv_dir: Optional[str]) -> str:
    if os.path.isabs(path_str):
        return path_str
    candidates = [path_str]
    if csv_dir:
        candidates.append(os.path.join(csv_dir, path_str))
    candidates.append(os.path.join(data_root, path_str))
    # Handle CheXpert-v1.0-small prefix by mapping to data_root.
    if "CheXpert-v1.0-small" in path_str:
        tail = path_str.split("CheXpert-v1.0-small", 1)[-1].lstrip(os.sep)
        candidates.append(os.path.join(data_root, tail))
    # If path contains a train/ or valid/ segment, map from there.
    for token in ("train/", "valid/"):
        if token in path_str:
            tail = path_str.split(token, 1)[-1]
            candidates.append(os.path.join(data_root, token.rstrip("/"), tail))
    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.normpath(cand)
    return os.path.normpath(os.path.join(data_root, path_str))


def _prepare_binary_labels(
    df: pd.DataFrame,
    label_cols: List[str],
    exclude_uncertain: bool,
) -> Tuple[pd.DataFrame, pd.Series]:
    labels = df[label_cols].apply(pd.to_numeric, errors="coerce")
    if exclude_uncertain:
        uncertain_mask = (labels == -1).any(axis=1)
        df = df.loc[~uncertain_mask].copy()
        labels = labels.loc[~uncertain_mask].copy()
    else:
        labels = labels.replace(-1, 0)
    labels = labels.fillna(0)

    other_cols = [c for c in label_cols if c != "No Finding"]
    no_finding = labels["No Finding"]
    normal_mask = (no_finding == 1) & (labels[other_cols].sum(axis=1) == 0)
    binary = (~normal_mask).astype(int)
    return df, binary


def load_chexpert_binary_dataframe(
    csv_path: str,
    data_root: str,
    exclude_uncertain: bool = True,
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    path_col = _find_path_column(df)
    df = _filter_frontal(df, path_col)
    label_cols = _infer_label_columns(df)
    df, binary = _prepare_binary_labels(df, label_cols, exclude_uncertain)
    df = df.copy()
    df["binary_label"] = binary.values
    csv_dir = os.path.dirname(os.path.abspath(csv_path))

    # Resolve paths and drop missing files defensively.
    resolved = []
    missing = 0
    for p in df[path_col].astype(str).tolist():
        full = _resolve_path(p, data_root, csv_dir)
        if not os.path.isfile(full):
            missing += 1
            resolved.append(None)
        else:
            resolved.append(full)
    df["resolved_path"] = resolved
    if missing:
        df = df[df["resolved_path"].notnull()].copy()
        print(f"Dropped {missing} rows with missing images.")
    return df, path_col


def _lung_focus_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    left = int(0.10 * w)
    right = int(0.90 * w)
    top = int(0.15 * h)
    bottom = int(0.95 * h)
    return img.crop((left, top, right, bottom))


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.Lambda(_lung_focus_crop),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class ChexpertBinaryDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
        lungmask_model: Optional[str] = None,
        lungmask_cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform or build_transforms()
        self.masker = LungMasker(model_path=lungmask_model, device=device)
        self.lungmask_cache_dir = lungmask_cache_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["resolved_path"]
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            # Fall back to a blank image if something is unreadable.
            img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
        if self.masker:
            cache_path = None
            if self.lungmask_cache_dir:
                key = hashlib.md5(img_path.encode("utf-8")).hexdigest()
                cache_path = os.path.join(self.lungmask_cache_dir, f"{key}.png")
            mask = cached_lung_mask(img, cache_path, self.masker)
            background = Image.new(img.mode, img.size, 0)
            img = Image.composite(img, background, mask)
        if self.transform:
            img = self.transform(img)
        label = int(row["binary_label"])
        return img, label, img_path
