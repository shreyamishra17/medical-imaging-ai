import argparse
import hashlib
import os

from PIL import Image

from data.chexpert_multilabel import load_chexpert_multilabel_dataframe
from data.lung_mask import LungMasker, cached_lung_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute lung masks for caching.")
    parser.add_argument("--data_root", default="datasets")
    parser.add_argument("--csv", default="datasets/train.csv")
    parser.add_argument("--out_dir", default="runs/lung_masks")
    parser.add_argument("--lungmask_model", default=None)
    parser.add_argument("--lungmask_device", default="cpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df, _ = load_chexpert_multilabel_dataframe(args.csv, args.data_root, exclude_uncertain=False)
    if args.limit:
        df = df.head(args.limit)

    masker = LungMasker(model_path=args.lungmask_model, device=args.lungmask_device)
    os.makedirs(args.out_dir, exist_ok=True)

    for idx, row in df.iterrows():
        path = row["resolved_path"]
        try:
            img = Image.open(path).convert("L")
        except Exception:
            continue
        key = hashlib.md5(path.encode("utf-8")).hexdigest()
        cache_path = os.path.join(args.out_dir, f"{key}.png")
        cached_lung_mask(img, cache_path, masker)

    print(f"Saved masks to {args.out_dir}")


if __name__ == "__main__":
    main()
