#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.chexpert_multilabel_infer import CheXpertMultiLabelPredictor

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACT_PATH = os.path.join(MODELS_DIR, "chexpert_multilabel_resnet18_logreg.joblib")
RESNET_W_PATH = os.path.join(MODELS_DIR, "resnet18-f37072fd.pth")

def score_to_likelihood(score: float) -> str:
    if score >= 0.85:
        return "Very likely"
    if score >= 0.70:
        return "Likely"
    if score >= 0.55:
        return "Possible"
    return "Unlikely"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", help="Path to a JPG/PNG X-ray image")
    ap.add_argument("--topk", type=int, default=5, help="Max findings to print")
    args = ap.parse_args()

    img_path = args.image_path
    if not os.path.isfile(img_path):
        raise SystemExit(f"Image not found: {img_path}")

    predictor = CheXpertMultiLabelPredictor(
        artifact_path=ARTIFACT_PATH,
        resnet_weights_path=RESNET_W_PATH,
        device=None,
    )

    img = Image.open(img_path)
    out = predictor.predict(img, topk_multi=args.topk)

    print("=" * 80)
    print("IMAGE:", img_path)
    print("FINAL:", out["final"])
    print("DEVICES PRESENT:", out["devices_present"])
    print("MODE:", out.get("mode", ""))
    note = out.get("primary_note", "")
    if note:
        print("NOTE:", note)

    if out["final"] == "NORMAL" or (out.get("has_abnormal_finding") is False):
        print("\nIMPRESSION (AI): No significant abnormality detected.")
        print("=" * 80)
        return

    primary = out["primary"]
    print("\nPRIMARY:")
    print(f"- {primary['label']} | score={primary['score']:.3f} | {score_to_likelihood(primary['score'])} | reliability={primary['reliability']}")

    print("\nFINDINGS SHOWN:")
    for f in out["findings"][:args.topk]:
        print(f"- {f['label']} | score={f['score']:.3f} | {score_to_likelihood(f['score'])} | reliability={f['reliability']}")

    print("\nALL LABEL SCORES (sorted):")
    for d in out["details"]:
        flag = "YES" if d["is_present"] else "NO"
        print(f"- {d['label']}: score={d['score']:.3f} present={flag} reliability={d['reliability']}")

    print("=" * 80)

if __name__ == "__main__":
    main()
