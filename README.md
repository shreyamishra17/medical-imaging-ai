# Medical Imaging & AI

A chest X-ray analysis system built as an honest, inspectable ML pipeline rather than a leaderboard-chasing classifier. It covers two phases: a binary pneumonia detector, and a multi-label CheXpert model (14 findings) with per-label threshold calibration, uncertainty-aware reporting, Grad-CAM explanations, and a Streamlit UI that serves patient-, clinician-, and engineer-facing views from the same inference engine.

The emphasis throughout is on **correctness and transparency over flashiness**: the system is explicit about what it does not know, refuses to claim a diagnosis when nothing crosses a calibrated threshold, and validates its predictions against ground truth inside the UI.

---

## Why this is more than a classifier

Most chest X-ray demos stop at "ResNet + 0.5 threshold + accuracy number." The harder, more realistic problems this project tackles:

- **A single X-ray has multiple simultaneous findings.** Binary classification is clinically unrealistic, so the core model is multi-label across 14 CheXpert findings (Pneumonia, Pleural Effusion, Edema, Cardiomegaly, Pneumothorax, and more).
- **The default 0.5 threshold is wrong for medical ML.** Class imbalance and label noise make a naive threshold fire false positives on normal images. This project measures how often each label falsely fires on *No Finding* images and sets a **per-label threshold** accordingly — so the model behaves like a cautious radiologist, not a spammy classifier.
- **Confidence without calibration is dangerous.** Each label carries a reliability badge derived from its validation AUROC, and the reporting layer uses hedged language ("most consistent with", "possible", "likely") instead of inventing anatomy or asserting causality.
- **Predictions are checked against ground truth.** The UI matches an uploaded image to the CheXpert validation set (by filename, size, and a partial content hash) and shows the overlap between model output and labeled truth — matched, missed, and extra findings — for instant sanity checking.

---

## Results

### Binary pneumonia (ResNet18 / DenseNet121)

Evaluated on a 624-image held-out test set. Because missing a pneumonia case is the costly error in screening, the models are tuned toward high recall on the positive class:

| Model | Accuracy | Pneumonia recall |
|---|---|---|
| ResNet18 (transfer learning) | ~88.5% | ~96.7% |
| DenseNet121 | ~86.7% | ~99.5% |

Confusion-matrix and false-negative analysis (in `reports/`) is used to weigh the recall–precision trade-off that matters most in screening, rather than optimizing accuracy alone.

### Multi-label CheXpert (14 findings)

A ResNet18 ImageNet backbone (frozen feature extractor) feeds a one-vs-rest logistic-regression classifier producing per-label probabilities. Per-label AUROC ranges roughly 0.6–0.75 — strong on well-represented findings, weaker on inherently hard, sparsely-labeled ones (Fracture, Lung Lesion), which is expected for CheXpert under weak supervision. The engineering value here is the **calibration and honesty layer** on top of the raw model, not a single headline metric.

---

## How it works

```
  X-ray image
      │
      ▼
  Preprocess  ──  grayscale → resize → ImageNet normalize
      │            (optional lung-mask preprocessing)
      ▼
  ResNet18 backbone  ──  512-d feature embedding
      │
      ▼
  Classifier  ──  binary head  OR  one-vs-rest multi-label
      │
      ▼
  Per-label thresholds  ──  calibrated on "No Finding" images
      │                      to suppress false positives
      ▼
  Decision layer  ──  NORMAL vs ABNORMAL · primary finding ·
      │                supporting findings · reliability badges
      ▼
  ┌─────────────────────┬──────────────────────┬─────────────────────┐
  │ Patient / clinician │  Ground-truth overlap │  Engineer view      │
  │ hedged report       │  (vs CheXpert labels) │  (scores, flags)    │
  └─────────────────────┴──────────────────────┴─────────────────────┘
                         Streamlit UI  +  Grad-CAM heatmaps
```

---

## Repository layout

```
src/
  chexpert_multilabel_infer.py   Inference engine: features → per-label thresholds → decision
  train_binary.py / eval_binary.py          Binary pneumonia training + evaluation
  train_multilabel.py / eval_multilabel.py  Multi-label CheXpert training + evaluation
  train_pneumonia.py / eval_pneumonia.py     Kermany pneumonia pipeline
  gradcam.py / gradcam_multilabel.py         Grad-CAM visual explanations
  precompute_lung_masks.py                   Optional lung-segmentation preprocessing
  validate_on_validcsv.py                    Offline validation against valid.csv
  data/                                      Dataset loaders (CheXpert, Kermany) + lung masks
  models/                                    Model builders (ResNet18 binary / multi-label)

app/
  streamlit_app.py    Patient / clinician / engineer UI with ground-truth overlay

analysis/             False-positive and true-positive-rate analysis scripts
scripts/              CLI prediction + legacy split/threshold-sweep tools
reports/              Saved evaluation reports (metrics only, no images)
notebooks/            Training notebook (historical, Kaggle)
PROJECT_OVERVIEW.md   Detailed design notes and project history
```

---

## Tech stack

PyTorch · torchvision (ResNet18, DenseNet121) · scikit-learn (one-vs-rest logistic regression, metrics) · OpenCV / Pillow (preprocessing) · Grad-CAM · Streamlit · NumPy / pandas

---

## Running it

The trained weights and datasets are not included (see below), so training or inference requires providing your own. Training, for example:

```bash
pip install -r requirements.txt
python src/train_multilabel.py --data_root datasets --epochs 3
python src/eval_multilabel.py  --checkpoint runs/multilabel_baseline/best.pt
```

The Streamlit app expects a trained artifact under `models/` and the CheXpert validation set under `datasets/`:

```bash
streamlit run app/streamlit_app.py
```

---

## Data & model weights are intentionally excluded

This repository contains **code only**. The datasets (CheXpert, Kermany pneumonia) and trained model weights are **not** committed, for two reasons:

- **Licensing** — CheXpert and the Kermany dataset carry redistribution-restricted licenses, so the images must be obtained directly from their sources, not re-hosted here.
- **Size & reproducibility** — model weights are large and regenerable from the training scripts.

This is a **research and prototyping system, not a medical device.** All outputs are AI-generated drafts meant to be correlated with clinical context.
