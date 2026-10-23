import os
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter
import torch


def _otsu_threshold(arr: np.ndarray) -> int:
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))
    total = arr.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    w_b = 0.0
    max_var = 0.0
    threshold = 128
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def _largest_components(mask: np.ndarray, k: int = 2, min_size: int = 800) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    sizes = {}
    label = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == 0:
                label += 1
                stack = [(y, x)]
                labels[y, x] = label
                count = 0
                while stack:
                    cy, cx = stack.pop()
                    count += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = label
                            stack.append((ny, nx))
                sizes[label] = count
    if not sizes:
        return mask
    keep = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:k]
    keep_labels = {lab for lab, size in keep if size >= min_size}
    if not keep_labels:
        keep_labels = {keep[0][0]}
    return np.isin(labels, list(keep_labels))


def _heuristic_mask(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("L"), dtype=np.uint8)
    h, w = arr.shape
    # Suppress likely text/marker regions before thresholding.
    arr[: int(0.20 * h), :] = 255
    arr[:, : int(0.08 * w)] = 255
    arr[:, int(0.92 * w) :] = 255
    thr = _otsu_threshold(arr)
    mask = arr < thr
    mask = _largest_components(mask, k=2, min_size=800)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))
    return mask_img


class LungMasker:
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.model = None
        self.device = torch.device(device)
        if model_path:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Lung mask model not found: {model_path}")
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()

    @torch.no_grad()
    def _model_mask(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)
        pred = self.model(x).squeeze().detach().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() + 1e-8)
        mask = (pred > 0.5).astype(np.uint8) * 255
        return Image.fromarray(mask)

    def get_mask(self, img: Image.Image) -> Image.Image:
        if self.model is None:
            return _heuristic_mask(img)
        return self._model_mask(img)

    def apply(self, img: Image.Image) -> Image.Image:
        mask = self.get_mask(img)
        background = Image.new(img.mode, img.size, 0)
        return Image.composite(img, background, mask)


def cached_lung_mask(
    img: Image.Image,
    cache_path: Optional[str],
    masker: "LungMasker",
) -> Image.Image:
    if cache_path and os.path.isfile(cache_path):
        return Image.open(cache_path).convert("L")
    mask = masker.get_mask(img)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        mask.save(cache_path)
    return mask
