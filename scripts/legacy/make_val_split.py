import os
import shutil
import random
import argparse

IMG_EXTS = (".jpg", ".jpeg", ".png")

def copy_files(files, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="Existing dataset root with train/ val/ test/")
    ap.add_argument("--dst_root", required=True, help="New dataset root to create with train/ val/ test/")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of TRAIN to copy into new VAL")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    src_train = os.path.join(args.src_root, "train")
    src_test  = os.path.join(args.src_root, "test")

    if not os.path.isdir(src_train):
        raise FileNotFoundError(f"Missing train folder: {src_train}")
    if not os.path.isdir(src_test):
        raise FileNotFoundError(f"Missing test folder: {src_test}")

    dst_train = os.path.join(args.dst_root, "train")
    dst_val   = os.path.join(args.dst_root, "val")
    dst_test  = os.path.join(args.dst_root, "test")

    if os.path.exists(args.dst_root):
        raise FileExistsError(f"{args.dst_root} already exists. Pick a new dst_root or delete it.")

    # 1) Copy TEST as-is (we keep test untouched)
    shutil.copytree(src_test, dst_test)

    # 2) Split TRAIN into new TRAIN + new VAL
    classes = [d for d in os.listdir(src_train) if os.path.isdir(os.path.join(src_train, d))]
    if not classes:
        raise RuntimeError(f"No class folders found inside {src_train}")

    print("Creating new split...")
    for cls in sorted(classes):
        cls_dir = os.path.join(src_train, cls)
        imgs = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS)]

        if len(imgs) == 0:
            raise RuntimeError(f"No images found for class {cls} in {cls_dir}")

        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * args.val_ratio))

        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        copy_files(train_imgs, os.path.join(dst_train, cls))
        copy_files(val_imgs, os.path.join(dst_val, cls))

        print(f"{cls}: total={len(imgs)}  new_train={len(train_imgs)}  new_val={len(val_imgs)}")

    print("\nDone.")
    print("New dataset root:", args.dst_root)
    print("Test copied as-is to:", dst_test)

if __name__ == "__main__":
    main()
