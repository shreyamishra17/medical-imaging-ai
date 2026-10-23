import os
import pandas as pd
from PIL import Image

from src.chexpert_multilabel_infer import CheXpertMultiLabelPredictor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
VALID_CSV = os.path.join(DATASET_ROOT, "valid.csv")
VALID_DIR = os.path.join(DATASET_ROOT, "valid")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACT_PATH = os.path.join(MODELS_DIR, "chexpert_multilabel_resnet18_logreg.joblib")
RESNET_W_PATH = os.path.join(MODELS_DIR, "resnet18-f37072fd.pth")

LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

def format_gt(row):
    """Show only labels that are positive (1)."""
    positives = []
    for l in LABELS:
        v = row.get(l, None)
        if pd.notna(v) and float(v) == 1.0:
            positives.append(l)
    return positives

def main(n=20):
    print("VALID_CSV:", VALID_CSV)
    print("VALID_DIR:", VALID_DIR)
    print("ARTIFACT:", ARTIFACT_PATH)
    print("RESNET:", RESNET_W_PATH)
    print("-" * 80)

    df = pd.read_csv(VALID_CSV)

    # CheXpert CSV column is typically "Path"
    if "Path" not in df.columns:
        raise ValueError(f"valid.csv missing 'Path' column. Columns: {list(df.columns)[:20]}")

    predictor = CheXpertMultiLabelPredictor(
        artifact_path=ARTIFACT_PATH,
        resnet_weights_path=RESNET_W_PATH,
        device=None,
    )

    # Sample first n rows (simple and deterministic)
    df = df.head(n)

    for i, row in df.iterrows():
        rel_path = row["Path"]

        # Make CSV paths work with your extracted folder structure
        # valid.csv paths can start with "CheXpert-v1.0-small/..."
        rel_path = rel_path.replace("CheXpert-v1.0-small/", "")

        img_path = os.path.join(DATASET_ROOT, rel_path)

        if not os.path.isfile(img_path):
            print(f"\n[{i}] MISSING IMAGE FILE:", img_path)
            continue

        img = Image.open(img_path)
        out = predictor.predict(img)
 
        gt_pos = format_gt(row)
        pred_primary = out["primary"]["label"]
        pred_primary_score = out["primary"]["score"]
        final = out["final"]
        shown = [f"{f['label']}({f['score']:.2f})" for f in out["findings"]]

        print(f"\n[{i}] {rel_path}")
        print("GT positives:", gt_pos if gt_pos else ["(none)"])
        print("Model FINAL:", final)
        print(f"Primary: {pred_primary} ({pred_primary_score:.3f})")
        print("Shown findings:", shown if shown else ["(none)"])

if __name__ == "__main__":
    main(n=20)
