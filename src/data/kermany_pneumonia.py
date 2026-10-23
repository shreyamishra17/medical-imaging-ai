import hashlib
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from data.lung_mask import LungMasker, cached_lung_mask

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class KermanyPneumoniaDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        lungmask_model: Optional[str] = None,
        lungmask_cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Missing dataset dir: {root_dir}")
        self.ds = datasets.ImageFolder(root_dir)
        self.transform = transform or build_transforms()
        self.masker = LungMasker(model_path=lungmask_model, device=device)
        self.lungmask_cache_dir = lungmask_cache_dir

        # Expect class names NORMAL, PNEUMONIA
        self.class_to_idx = self.ds.class_to_idx
        if "NORMAL" not in self.class_to_idx or "PNEUMONIA" not in self.class_to_idx:
            raise ValueError(f"Expected NORMAL and PNEUMONIA folders in {root_dir}")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        path, label = self.ds.samples[idx]
        try:
            img = Image.open(path).convert("L")
        except Exception:
            img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))

        if self.masker:
            cache_path = None
            if self.lungmask_cache_dir:
                key = hashlib.md5(path.encode("utf-8")).hexdigest()
                cache_path = os.path.join(self.lungmask_cache_dir, f"{key}.png")
            mask = cached_lung_mask(img, cache_path, self.masker)
            background = Image.new(img.mode, img.size, 0)
            img = Image.composite(img, background, mask)

        if self.transform:
            img = self.transform(img)
        return img, int(label), path
