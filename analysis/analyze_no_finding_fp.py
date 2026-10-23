import os
import numpy as np
import pandas as pd
from PIL import Image

from src.chexpert_multilabel_infer import CheXpertMultiLabelPredictor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
VALID_CSV = os.path.join(DATASET_ROOT, "valid.csv")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACT_PATH = os.path.join(MODELS_DIR, "chexpert_multilabel_resnet18_logreg.joblib")
RESNET_W_PATH = os.path.join(MODELS_DIR, "resnet18-f37072fd.pth")

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

DISEASE_LABELS = [l for l in LABELS if l not in ("No Finding", "Support Devices")]

def resolve_path(rel_path: str) -> str:
    # valid.csv often stores paths like "CheXpert-v1.0-small/valid/..."
    rel_path = rel_path.replace("CheXpert-v1.0-small/", "")
    return os.path.join(DATASET_ROOT, rel_path)

def main(max_images=400, score_floor=0.0):
    df = pd.read_csv(VALID_CSV)
    if "Path" not in df.columns:
        raise ValueError("valid.csv missing Path column")

    # Keep only rows that are confidently "No Finding" positive (1)
    # (Ignore uncertain/missing)
    nf = df[df["No Finding"] == 1.0].copy()
    nf = nf.head(max_images)  # keep it fast for first run

    predictor = CheXpertMultiLabelPredictor(
        artifact_path=ARTIFACT_PATH,
        resnet_weights_path=RESNET_W_PATH,
        device=None,
    )

    # Collect scores by label on No Finding images
    scores = {l: [] for l in DISEASE_LABELS}

    n_used = 0
    for _, row in nf.iterrows():
        img_path = resolve_path(row["Path"])
        if not os.path.isfile(img_path):
            continue

        img = Image.open(img_path)
        out = predictor.predict(img)

        # Use out["details"] which includes score + is_present for all labels except No Finding
        # We'll capture raw scores (not just thresholded)
        detail_map = {d["label"]: d["score"] for d in out["details"]}

        for l in DISEASE_LABELS:
            s = float(detail_map.get(l, 0.0))
            if s >= score_floor:
                scores[l].append(s)

        n_used += 1

    print(f"Used {n_used} 'No Finding' images (max_images={max_images}).")

    # Summarize: how often each disease exceeds current threshold, plus score stats
    rows = []
    for l in DISEASE_LABELS:
        arr = np.array(scores[l], dtype=float)
        if arr.size == 0:
            continue
        thr = float(predictor.thresholds.get(l, 0.5))
        fire_rate = float((arr >= thr).mean())
        rows.append({
            "label": l,
            "threshold": thr,
            "fires_on_no_finding_%": round(100 * fire_rate, 2),
            "p50_score": round(float(np.percentile(arr, 50)), 3),
            "p90_score": round(float(np.percentile(arr, 90)), 3),
            "max_score": round(float(arr.max()), 3),
            "n": int(arr.size),
        })

    outdf = pd.DataFrame(rows).sort_values("fires_on_no_finding_%", ascending=False)
    print("\nFalse-positive pressure on No Finding (higher is worse):")
    print(outdf.to_string(index=False))

if __name__ == "__main__":
    main(max_images=400)
