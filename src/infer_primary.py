import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from data.chexpert_multilabel import build_transforms
from data.lung_mask import LungMasker
from models.resnet18_multilabel import build_resnet18_multilabel, load_checkpoint


def _select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _reliability_badge(label_auc, label: str) -> str:
    auc = label_auc.get(label, None)
    if auc is None:
        return "Unknown"
    if auc >= 0.75:
        return "High"
    if auc >= 0.65:
        return "Medium"
    return "Low"


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer primary impression from a multi-label model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lungmask_model", default=None)
    parser.add_argument("--lungmask_device", default="cpu")
    parser.add_argument("--single_gap", type=float, default=0.12)
    parser.add_argument("--min_primary_score", type=float, default=0.5)
    args = parser.parse_args()

    device = _select_device(args.device)
    model = build_resnet18_multilabel(num_labels=14, pretrained=False).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, device)
    label_names = checkpoint.get("label_names")
    if label_names is None:
        raise ValueError("Checkpoint missing label_names; retrain or update checkpoint.")
    if len(label_names) != model.fc.out_features:
        model = build_resnet18_multilabel(num_labels=len(label_names), pretrained=False).to(device)
        load_checkpoint(model, args.checkpoint, device)
    model.eval()

    thresholds = {name: 0.5 for name in label_names}
    thresholds_path = args.thresholds
    if thresholds_path is None:
        thresholds_path = os.path.join(os.path.dirname(args.checkpoint), "thresholds.json")
    if os.path.isfile(thresholds_path):
        with open(thresholds_path, "r") as f:
            thresholds.update(json.load(f))

    img = Image.open(args.image).convert("L")
    masker = LungMasker(model_path=args.lungmask_model, device=args.lungmask_device)
    img = masker.apply(img)
    x = build_transforms()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0).cpu().numpy().astype(float)

    label_auc = checkpoint.get("label_auc", {})
    # Conservative gating: require higher thresholds for low-reliability labels.
    thresholds_adj = {}
    for name in label_names:
        base = float(thresholds.get(name, 0.5))
        rel = _reliability_badge(label_auc, name)
        if rel == "Medium":
            base = min(0.95, base + 0.05)
        elif rel == "Low":
            base = min(0.95, base + 0.15)
        thresholds_adj[name] = base

    rows = []
    for name, prob in zip(label_names, probs):
        rel = _reliability_badge(label_auc, name)
        yes = int(prob >= float(thresholds_adj.get(name, 0.5)))
        rows.append((name, float(prob), yes, rel))

    disease_labels = set(label_names) - {"No Finding", "Support Devices"}
    disease_present = any((name in disease_labels and yes == 1) for (name, _, yes, _) in rows)
    final = "ABNORMAL" if disease_present else "NORMAL"

    dev_row = next((r for r in rows if r[0] == "Support Devices"), None)
    devices_present = "Yes" if (dev_row and dev_row[2] == 1) else "No"

    exclude_primary = {"No Finding", "Support Devices"}
    present_candidates = [r for r in rows if (r[0] not in exclude_primary and r[2] == 1)]
    ranked_all = sorted(
        [r for r in rows if r[0] not in exclude_primary],
        key=lambda x: x[1],
        reverse=True,
    )

    if present_candidates:
        ranked_present = sorted(present_candidates, key=lambda x: x[1], reverse=True)
        primary = ranked_present[0]
        second = ranked_present[1] if len(ranked_present) > 1 else None
        close_gap = (second is not None and (primary[1] - second[1]) < 0.05)
        primary_note = "Close call (top two are similar)" if close_gap else ""
    else:
        primary = ranked_all[0] if ranked_all else ("(none)", 0.0, 0, "Unknown")
        primary_note = "No finding crossed thresholds"

    if primary[1] < args.min_primary_score:
        primary_note = "Low confidence"

    if present_candidates and len(present_candidates) >= 2:
        gap = present_candidates[0][1] - present_candidates[1][1]
    elif len(ranked_all) >= 2:
        gap = ranked_all[0][1] - ranked_all[1][1]
    else:
        gap = 1.0

    single_issue = (final == "ABNORMAL") and (gap >= args.single_gap) and (len(present_candidates) > 0)

    exclude_display = {"No Finding", "Support Devices"}
    display = [r for r in rows if r[0] not in exclude_display and r[2] == 1]
    display = sorted(display, key=lambda x: x[1], reverse=True)

    if final == "NORMAL":
        concise = []
    else:
        if single_issue:
            supports = [r for r in display if r[0] != primary[0]][:2]
            concise = [primary] + supports
        else:
            concise = display[:5]
            if primary[0] not in [r[0] for r in concise]:
                concise = [primary] + concise
                concise = concise[:5]

    details = sorted([r for r in rows if r[0] != "No Finding"], key=lambda x: x[1], reverse=True)
    has_abnormal_finding = len(present_candidates) > 0

    result = {
        "final": final,
        "primary": {"label": primary[0], "score": primary[1], "reliability": primary[3]},
        "devices_present": devices_present,
        "primary_note": primary_note,
        "has_abnormal_finding": has_abnormal_finding,
        "mode": "Single-issue" if single_issue else "Multi-issue",
        "findings": [{"label": n, "score": p, "reliability": rel} for (n, p, y, rel) in concise],
        "details": [{"label": n, "score": p, "is_present": bool(y), "reliability": rel} for (n, p, y, rel) in details],
    }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.out_json}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
