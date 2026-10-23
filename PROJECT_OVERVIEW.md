# Medical Imaging & AI -- Project Overview (Handoff Summary)

## 1. High-level vision (big picture)

The long-term goal of this project is to build a general medical imaging AI platform that can:

* Work across modalities (X-ray -> CT -> MRI in the future)
* Support multiple diseases per image (not just single-disease classification)
* Produce human-readable, clinician- and patient-friendly reports
* Be inspectable, explainable, and honest about uncertainty
* Serve as a research / prototyping system, not a regulated medical product

Right now, the project is focused on Chest X-rays as the starting modality, with CheXpert as the primary dataset.

---

## 2. Folder structure (current state)

```
medical_imaging_ai/
|
|-- app/
|   |-- streamlit_app.py        # Main UI (patient + engineer views)
|
|-- src/
|   |-- chexpert_multilabel_infer.py  # Inference engine (model + thresholds)
|   |-- validate_on_validcsv.py       # Offline validation against valid.csv
|   |-- __init__.py
|
|-- models/
|   |-- chexpert_multilabel_resnet18_logreg.joblib  # Trained multilabel model
|   |-- resnet18-f37072fd.pth                        # ImageNet weights
|   |-- threshold.json                               # Per-label tuned thresholds
|
|-- datasets/
|   |-- train/              # CheXpert train images
|   |-- valid/              # CheXpert valid images
|   |-- train.csv
|   |-- valid.csv
|
|-- scripts/
|   |-- predict_image.py    # CLI prediction tool (legacy / debugging)
|
|-- kaggle/
|   |-- notebooks used for training (historical)
```

---

## 3. Phase 1 -- Binary Pneumonia Model (historical)

### What was done

* Initial experimentation used a binary pneumonia classifier
* Model: ResNet18
* Dataset: smaller pneumonia-focused dataset
* Achieved ~89% accuracy
* This validated:

  * End-to-end training pipeline
  * Image preprocessing
  * Model export + local inference

### Why we moved on

* Binary classification is not clinically realistic
* Real chest X-rays have multiple simultaneous findings
* Accuracy alone is misleading in medical imaging

This phase is complete and archived conceptually.

---

## 4. Phase 2 -- Multilabel CheXpert Model (current core)

### Dataset

* CheXpert-v1.0-small
* 224,000+ images
* 14 labels, multi-label, weakly supervised
* Labels include:

  * Pneumonia, Pneumothorax, Pleural Effusion, Edema, Lung Opacity, etc.
* Labels can be:

  * 1 = positive
  * 0 = negative
  * -1 = uncertain
  * NaN = not mentioned

### Key realization

* Filenames mean nothing
* valid.csv / train.csv are the only ground truth
* A single image can have 5-6 true labels

---

## 5. Training strategy (constraints-aware)

### Why this approach

* No GPU on Kaggle (CPU only)
* Need fast iteration
* Goal is correctness + interpretability, not SOTA leaderboard scores

### Model architecture

* ResNet18 (ImageNet pretrained) -> frozen feature extractor
* Output: 512-d embedding
* Classifier:

  * OneVsRest Logistic Regression
  * Multi-label probability output

### Training outcome

* AUROC varies by label (~0.6-0.75)
* Some labels are inherently weak (e.g., Fracture, Lung Lesion)
* This is expected for CheXpert + weak supervision

### Artifact saved

chexpert_multilabel_resnet18_logreg.joblib contains:

* Trained classifier
* Label list
* Per-label AUROC
* Default thresholds (later overridden)

---

## 6. Major correctness issue discovered (and fixed)

### Problem

* Model was firing diseases on No Finding images
* Even visually normal images triggered labels like:

  * Fracture
  * Lung Lesion
  * Enlarged Cardiomediastinum

### Why this happens

* CheXpert has:

  * Label noise
  * Uncertain labels
  * Strong class imbalance
* Default threshold = 0.5 is wrong for medical ML

---

## 7. Threshold calibration (critical step)

### What we did

* Used only No Finding images from valid.csv
* Measured:

  * How often each label falsely "fires" on normal images
* Built a table:

  * fires_on_no_finding_%
  * Score distributions (p50, p90, max)

### Result

* Created per-label thresholds
* Example:

  * Fracture -> threshold ~0.90
  * Lung Opacity -> threshold ~0.48
  * Pleural Effusion -> threshold ~0.50

These thresholds are saved in:

```
models/threshold.json
```

### Effect

* Dramatically reduced false positives
* Made predictions defensible
* Model now behaves like a cautious radiologist, not a spammy classifier

---

## 8. Inference engine (chexpert_multilabel_infer.py)

### Responsibilities

* Load model + thresholds
* Extract features
* Apply per-label thresholds
* Decide:

  * NORMAL vs ABNORMAL
  * Primary finding
  * Supporting findings
* Attach reliability levels using AUROC

### Key design decisions

* Never claim a diagnosis if nothing crosses thresholds
* Separate:

  * Support Devices (informational)
  * No Finding (special case)
* Support single-issue vs multi-issue logic
* Output structured dict for UI

This file is the single source of truth for inference.

---

## 9. Streamlit UI (current state)

### Goals

* One UI for everyone
* Different layers for:

  * Patient / non-clinical users
  * Clinicians
  * Engineers

### Current structure

* Tab 1: Upload any image

  * Input image
  * Dropdown sections:

    1. Patient / Clinician-friendly report (default open)
    2. Ground Truth (CheXpert valid.csv, hidden)
    3. ML / Engineer view (hidden)

* Tab 2: CheXpert valid

  * Randomly sample known images
  * Always shows ground truth
  * Used for sanity checks and debugging

---

## 10. Patient / Clinician-friendly report (important)

### Why this exists

* Raw probabilities are useless to humans
* Clinicians read Findings + Impression, not logits

### What we do (intentionally limited)

* DO:

  * "Findings are most consistent with..."
  * "Possible / Likely" phrasing
  * Explicit uncertainty
* DO NOT:

  * Invent anatomy ("right lower lobe")
  * Claim causality
  * Hide uncertainty

This keeps the system ethical and honest.

---

## 11. Ground Truth integration (key feature)

### How GT works

* If uploaded image exists in datasets/valid/
* Match by:

  * Filename
  * File size
  * Partial SHA-1 hash
* Automatically pull labels from valid.csv

### UI shows

* GT positive labels
* Uncertain labels
* Overlap summary:

  * Matched
  * Missed
  * Extra

This allows instant sanity-checking inside the UI, without manual CSV work.

---

## 12. Where the project is right now

We currently have:

* A working multilabel model
* Threshold-calibrated predictions
* Honest, readable reports
* Integrated ground truth validation
* Clean separation of inference vs UI

We do not yet have:

* Localization (Grad-CAM, bounding boxes)
* Deep CNN finetuning (DenseNet on GPU)
* CT or MRI support
* Regulatory-grade performance

And that's intentional.

---

## 13. Next logical steps (future)

Short-term:

* Add Grad-CAM visual explanations
* Retrain on GPU with DenseNet121
* Improve weak labels handling (-1 strategy)

Mid-term:

* CT scan pipeline
* Slice aggregation
* Structured radiology report generation

Long-term:

* Unified imaging platform
* Multimodal inputs
* Hospital-scale workflows

---

## 14. Design philosophy (important)

This project prioritizes:

* Correctness over flashiness
* Transparency over confidence
* Human readability
* Engineering sanity

Every choice so far reflects that.
