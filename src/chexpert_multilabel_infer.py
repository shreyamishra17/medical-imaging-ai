import os
import joblib
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

DEFAULT_LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

class CheXpertMultiLabelPredictor:
    def __init__(self, artifact_path: str, resnet_weights_path: str, device: str = None):
        if not os.path.isfile(artifact_path):
            raise FileNotFoundError(f"Missing artifact: {artifact_path}")
        if not os.path.isfile(resnet_weights_path):
            raise FileNotFoundError(f"Missing resnet weights: {resnet_weights_path}")

        self.art = joblib.load(artifact_path)
        self.clf = self.art["model"]
        self.labels = self.art.get("labels", DEFAULT_LABELS)
        # Base thresholds from training artifact
        base_thresholds = self.art.get("thresholds", {l: 0.5 for l in self.labels})

        # Optional override thresholds from JSON (calibration)
        thresholds_path = os.path.join(os.path.dirname(artifact_path), "thresholds.json")
        if os.path.isfile(thresholds_path):
            with open(thresholds_path, "r") as f:
                override_thresholds = json.load(f)
            base_thresholds.update(override_thresholds)

        self.thresholds = base_thresholds

        self.label_auc = self.art.get("label_auc", {})

        # defaults (match your Kaggle pipeline)
        self.img_size = int(self.art.get("img_size", 160))
        self.mean = self.art.get("imagenet_mean", [0.485, 0.456, 0.406])
        self.std = self.art.get("imagenet_std", [0.229, 0.224, 0.225])

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)

        # ResNet18 feature extractor -> 512-d
        net = resnet18(weights=None)
        state = torch.load(resnet_weights_path, map_location="cpu")
        net.load_state_dict(state)
        net.fc = nn.Identity()
        net.eval()
        net.to(self.device)
        self.feat_net = net

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    @torch.no_grad()
    def _extract_feature(self, pil_img: Image.Image) -> np.ndarray:
        img = pil_img.convert("L")
        x = self.transform(img).unsqueeze(0).to(self.device)  # (1,3,H,W)
        f = self.feat_net(x)  # (1,512)
        return f.cpu().numpy().astype(np.float32)

    def _reliability_badge(self, label: str) -> str:
        auc = self.label_auc.get(label, None)
        if auc is None:
            return "Unknown"
        if auc >= 0.75:
            return "High"
        if auc >= 0.65:
            return "Medium"
        return "Low"

    def predict(self, pil_img: Image.Image,
                topk_multi: int = 5,
                max_support_single: int = 2,
                single_gap: float = 0.12):

        X = self._extract_feature(pil_img)   # (1,512)
        P = self.clf.predict_proba(X)[0]     # (14,)

        rows = []
        for name, prob in zip(self.labels, P):
            prob = float(prob)
            yes = int(prob >= float(self.thresholds.get(name, 0.5)))
            rel = self._reliability_badge(name)
            rows.append((name, prob, yes, rel))
        DISEASE_LABELS = [l for l in self.labels if l not in ("No Finding", "Support Devices")]
        disease_present = any((name in DISEASE_LABELS and yes == 1) for (name, prob, yes, rel) in rows)

        final = "ABNORMAL" if disease_present else "NORMAL"

        # NORMAL / ABNORMAL based on disease-positive predictions
        # A study is abnormal if any disease label crosses threshold
        DISEASE_LABELS = set(self.labels) - {"No Finding", "Support Devices"}
        disease_present = any((name in DISEASE_LABELS and yes == 1) for (name, prob, yes, rel) in rows)

        final = "ABNORMAL" if disease_present else "NORMAL"

        # Devices present (Support Devices -> yes/no)
        dev_row = next((r for r in rows if r[0] == "Support Devices"), None)
        devices_present = "Yes" if (dev_row and dev_row[2] == 1) else "No"

        # Primary finding: only choose from labels that crossed thresholds
        exclude_primary = {"No Finding", "Support Devices"}

        present_candidates = [
            r for r in rows
            if (r[0] not in exclude_primary and r[2] == 1)
        ]

        ranked_all = sorted(
            [r for r in rows if r[0] not in exclude_primary],
            key=lambda x: x[1],
            reverse=True
        )

        if present_candidates:
            ranked_present = sorted(present_candidates, key=lambda x: x[1], reverse=True)
            top = ranked_present[0]
            second = ranked_present[1] if len(ranked_present) > 1 else None

            CLOSE_GAP = 0.05
            primary = top
            primary_note = "Close call (top two are similar)" if (second and (top[1] - second[1]) < CLOSE_GAP) else ""
        else:
            # Nothing crossed thresholds: do NOT claim a diagnosis in clinician view
            primary = ranked_all[0] if ranked_all else ("(none)", 0.0, 0, "Unknown")
            primary_note = "No finding crossed thresholds (debug top score shown)"

        # Single vs multi: use gap among PRESENT findings if available, else fall back to top scores
        if present_candidates and len(present_candidates) >= 2:
            ranked_present = sorted(present_candidates, key=lambda x: x[1], reverse=True)
            gap = ranked_present[0][1] - ranked_present[1][1]
        elif ranked_all and len(ranked_all) >= 2:
            gap = ranked_all[0][1] - ranked_all[1][1]
        else:
            gap = 1.0

        single_issue = (final == "ABNORMAL") and (gap >= single_gap) and (len(present_candidates) > 0)

        # UI list excludes Support Devices (we show it separately)
        exclude_display = {"No Finding", "Support Devices"}
        display = [r for r in rows if r[0] not in exclude_display and r[2] == 1]
        display = sorted(display, key=lambda x: x[1], reverse=True)

        if final == "NORMAL":
            concise = []
        else:
            if single_issue:
                supports = [r for r in display if r[0] != primary[0]][:max_support_single]
                concise = [primary] + supports
            else:
                concise = display[:topk_multi]
                if primary[0] not in [r[0] for r in concise]:
                    concise = [primary] + concise
                    concise = concise[:topk_multi]

        details = sorted([r for r in rows if r[0] != "No Finding"], key=lambda x: x[1], reverse=True)
        has_abnormal_finding = len(present_candidates) > 0

        return {
            "final": final,
            "primary": {"label": primary[0], "score": primary[1], "reliability": primary[3]},
            "devices_present": devices_present,
            "primary_note": primary_note,
            "has_abnormal_finding": has_abnormal_finding,
            "mode": "Single-issue" if single_issue else "Multi-issue",
            "findings": [{"label": n, "score": p, "reliability": rel} for (n, p, y, rel) in concise],
            "details": [{"label": n, "score": p, "is_present": bool(y), "reliability": rel} for (n, p, y, rel) in details],
        }
        