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
    rel_path = rel_path.replace("CheXpert-v1.0-small/", "")
    return os.path.join(DATASET_ROOT, rel_path)

def main(max_per_label=75):
    df = pd.read_csv(VALID_CSV)
    predictor = CheXpertMultiLabelPredictor(
        artifact_path=ARTIFACT_PATH,
        resnet_weights_path=RESNET_W_PATH,
        device=None,
    )

    results = []

    for label in DISEASE_LABELS:
        if label not in df.columns:
            continue

        # Rows where GT is confidently positive (==1)
        sub = df[df[label] == 1.0].copy().head(max_per_label)

        used = 0
        fired = 0
        scores = []

        for _, row in sub.iterrows():
            img_path = resolve_path(row["Path"])
            if not os.path.isfile(img_path):
                continue

            img = Image.open(img_path)
            out = predictor.predict(img)
            detail_map = {d["label"]: d for d in out["details"]}

            if label not in detail_map:
                continue

            s = float(detail_map[label]["score"])
            is_present = bool(detail_map[label]["is_present"])
            scores.append(s)

            used += 1
            if is_present:
                fired += 1

        if used == 0:
            continue

        arr = np.array(scores, dtype=float)
        results.append({
            "label": label,
            "threshold": float(predictor.thresholds.get(label, 0.5)),
            "tp_fire_%": round(100 * fired / used, 2),  # how often we detect GT positives
            "p50_score": round(float(np.percentile(arr, 50)), 3),
            "p10_score": round(float(np.percentile(arr, 10)), 3),
            "max_score": round(float(arr.max()), 3),
            "n_pos_used": used
        })

    outdf = pd.DataFrame(results).sort_values("tp_fire_%", ascending=True)
    print("\nTrue-positive detection rate on valid.csv (higher is better):")
    print(outdf.to_string(index=False))

if __name__ == "__main__":
    main(max_per_label=75)
