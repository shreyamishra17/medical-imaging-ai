# app/streamlit_app.py
import os
import sys
import pandas as pd
import streamlit as st
from PIL import Image
import hashlib
from pathlib import Path

# Ensure project root is on Python path so `import src...` works
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.chexpert_multilabel_infer import CheXpertMultiLabelPredictor

# -------------------------
# Paths
# -------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
ARTIFACT_PATH = os.path.join(MODELS_DIR, "chexpert_multilabel_resnet18_logreg.joblib")
RESNET_W_PATH = os.path.join(MODELS_DIR, "resnet18-f37072fd.pth")

DATASET_DIR = os.path.join(PROJECT_ROOT, "datasets")
VALID_CSV_PATH = os.path.join(DATASET_DIR, "valid.csv")

# -------------------------
# Labels
# -------------------------
LABELS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

# -------------------------
# ML helper text mapping (engineer-ish)
# -------------------------
def score_to_likelihood(score: float) -> str:
    if score >= 0.85:
        return "Very likely"
    if score >= 0.70:
        return "Likely"
    if score >= 0.55:
        return "Possible"
    return "Unlikely"

def reliability_to_text(rel: str) -> str:
    if rel == "High":
        return "High"
    if rel == "Medium":
        return "Moderate"
    if rel == "Low":
        return "Low"
    return "Unknown"

# -------------------------
# Valid.csv helpers
# -------------------------
@st.cache_data
def load_valid_df():
    if not os.path.isfile(VALID_CSV_PATH):
        return None
    df = pd.read_csv(VALID_CSV_PATH)
    # Normalize to match your local extracted dataset folder structure
    df["Path_norm"] = df["Path"].astype(str).str.replace("CheXpert-v1.0-small/", "", regex=False)
    return df

def gt_from_row(row):
    gt_pos, gt_unc = [], []
    for l in LABELS:
        v = row.get(l, None)
        if pd.isna(v):
            continue
        v = float(v)
        if v == 1.0:
            gt_pos.append(l)
        elif v == -1.0:
            gt_unc.append(l)
    return gt_pos, gt_unc

@st.cache_resource
def build_valid_file_index():
    """
    Build an index of files in datasets/valid:
      key = (filename, filesize, sha1_prefix)
      value = relative path like valid/patient.../view1_frontal.jpg

    Uses sha1 prefix (first 12 hex chars) for speed.
    """
    valid_root = Path(DATASET_DIR) / "valid"
    if not valid_root.exists():
        return {}

    index = {}
    for p in valid_root.rglob("*.jpg"):
        try:
            size = p.stat().st_size
            h = hashlib.sha1()
            with open(p, "rb") as f:
                h.update(f.read(256 * 1024))  # first 256KB
            sha1p = h.hexdigest()[:12]
            index[(p.name, size, sha1p)] = str(p.relative_to(Path(DATASET_DIR)))
        except Exception:
            continue
    return index

def hash_upload(file_bytes: bytes) -> str:
    h = hashlib.sha1()
    h.update(file_bytes[:256 * 1024])
    return h.hexdigest()[:12]

def find_gt_for_uploaded(uploaded_file, valid_df, valid_index):
    """
    Returns (gt_pos, gt_unc, matched_rel_path) or (None, None, None)
    """
    file_bytes = uploaded_file.getvalue()
    fname = uploaded_file.name
    size = len(file_bytes)
    sha1p = hash_upload(file_bytes)

    rel_path = valid_index.get((fname, size, sha1p))
    if not rel_path:
        return None, None, None

    # rel_path like "valid/patient.../view1_frontal.jpg"
    m = valid_df[valid_df["Path_norm"].astype(str).str.endswith(rel_path)]
    if len(m) == 0:
        return None, None, rel_path

    row = m.iloc[0]
    gt_pos, gt_unc = gt_from_row(row)
    return gt_pos, gt_unc, rel_path

# -------------------------
# Patient/Clinician-friendly report helpers
# -------------------------
def friendly_name(label: str) -> str:
    mapping = {
        "Pleural Effusion": "Fluid around the lung (pleural effusion)",
        "Lung Opacity": "Lung haziness / opacity",
        "Edema": "Fluid-related lung pattern (edema)",
        "Atelectasis": "Small areas of lung under-expansion (atelectasis)",
        "Pneumothorax": "Air outside the lung (pneumothorax)",
        "Consolidation": "Denser lung pattern (consolidation)",
        "Pneumonia": "Infectious-type pattern (pneumonia label)",
        "Cardiomegaly": "Enlarged heart silhouette (cardiomegaly)",
        "Enlarged Cardiomediastinum": "Widened heart/mediastinum contour",
        "Lung Lesion": "Focal lung lesion label",
        "Pleural Other": "Other pleural finding label",
        "Fracture": "Possible bone injury (fracture label)",
        "Support Devices": "Support devices present",
        "No Finding": "No significant abnormality detected",
    }
    return mapping.get(label, label)

def confidence_phrase(score: float) -> str:
    if score >= 0.85:
        return "more likely"
    if score >= 0.70:
        return "likely"
    if score >= 0.55:
        return "possible"
    return "less likely"

def build_patient_report(out: dict) -> dict:
    """
    Produces a patient/clinician-friendly "Findings / Impression" report.
    This does NOT invent anatomy or causal explanations.
    """
    final = out.get("final", "ABNORMAL")
    devices = out.get("devices_present", "Unknown")
    shown = out.get("findings", [])
    primary = out.get("primary", {"label": "(none)", "score": 0.0, "reliability": "Unknown"})

    # NORMAL or nothing crossed thresholds
    if final == "NORMAL" or out.get("has_abnormal_finding") is False or len(shown) == 0:
        md = (
            "### Findings\n"
            "- No significant abnormality flagged by the model.\n\n"
            "### Impression\n"
            "- **Overall: No major abnormality detected** (AI draft).\n\n"
            f"**Support devices:** {devices}\n\n"
            "_Note: This is an AI-generated summary for research use and should be correlated with clinical context._"
        )
        return {"status": "normal", "markdown": md, "key_findings": []}

    # ABNORMAL
    primary_label = primary["label"]
    primary_score = float(primary["score"])
    primary_rel = primary.get("reliability", "Unknown")

    bullets = []
    for f in shown[:4]:
        lbl = f["label"]
        sc = float(f["score"])
        bullets.append(f"- **{friendly_name(lbl)}** — {confidence_phrase(sc)}")

    findings_txt = "\n".join(bullets) if bullets else "- None highlighted"

    note = out.get("primary_note", "")
    note_line = f"\n\n**Model note:** {note}" if note else ""

    md = (
        "### Findings\n"
        f"{findings_txt}\n\n"
        "### Impression\n"
        f"- **Overall: Abnormal study** (AI draft)\n"
        f"- **Most consistent with:** **{friendly_name(primary_label)}** — {confidence_phrase(primary_score)}\n"
        f"- **Model reliability for this label:** {primary_rel}\n"
        f"- **Support devices:** {devices}"
        f"{note_line}\n\n"
        "_Note: This is an AI-generated summary for research use and should be correlated with symptoms, history, and prior imaging._"
    )

    return {
        "status": "abnormal",
        "markdown": md,
        "key_findings": [f["label"] for f in shown]
    }

# -------------------------
# GT overlap summary helpers
# -------------------------
def predicted_present_labels(out: dict) -> list:
    """
    Uses the predictor's thresholded 'is_present' flags from details.
    Excludes 'No Finding' since it's not a 'finding' in the same sense.
    """
    labels = []
    for d in out.get("details", []):
        if d.get("label") == "No Finding":
            continue
        if bool(d.get("is_present")):
            labels.append(d.get("label"))
    return labels

def render_gt_overlap(out: dict, gt_pos: list):
    pred = set(predicted_present_labels(out))
    gt = set(gt_pos)

    matched = sorted(list(pred.intersection(gt)))
    missed = sorted(list(gt.difference(pred)))
    extra = sorted(list(pred.difference(gt)))

    if len(gt) == 0 and len(pred) == 0:
        st.success("✅ GT overlap: Both GT and model show no positive findings.")
        return

    # Simple badge
    if len(matched) > 0:
        st.success(f"✅ Matched: {', '.join(matched)}" if matched else "✅ Matched: (none)")
    else:
        st.warning("⚠️ Matched: (none)")

    if missed:
        st.warning("Missed GT labels: " + ", ".join(missed))
    else:
        st.caption("Missed GT labels: (none)")

    if extra:
        st.info("Extra model labels (not in GT): " + ", ".join(extra))
    else:
        st.caption("Extra model labels: (none)")

# -------------------------
# Predictor
# -------------------------
@st.cache_resource
def load_predictor():
    return CheXpertMultiLabelPredictor(
        artifact_path=ARTIFACT_PATH,
        resnet_weights_path=RESNET_W_PATH,
        device=None,
    )

predictor = load_predictor()

# -------------------------
# Streamlit page
# -------------------------
st.set_page_config(page_title="Chest X-ray Findings (Local)", layout="centered")
st.title("Chest X-ray Findings (Local)")
st.caption("Research/demo tool — not a medical device.")

tab_upload, tab_valid = st.tabs(["Upload any image", "CheXpert valid (Ground Truth)"])

# =========================
# TAB 1: Upload any image (optional GT if match)
# =========================
with tab_upload:
    uploaded = st.file_uploader("Upload an X-ray image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to get predictions.")
    else:
        img = Image.open(uploaded)

        st.subheader("Input image")
        st.image(img, use_container_width=True)

        out = predictor.predict(img)

        # Build GT lookup tools (only once)
        valid_df = load_valid_df()
        valid_index = build_valid_file_index() if valid_df is not None else {}

        st.subheader("Results")

        # 1) Patient / Clinician report (default open)
        report = build_patient_report(out)
        with st.expander("Clinician / Patient-friendly report", expanded=True):
            if report["status"] == "normal":
                st.success("IMPRESSION: No significant abnormality detected (AI draft)")
            else:
                st.warning("IMPRESSION: Abnormal study (AI draft)")
            st.markdown(report["markdown"])

        # 2) Ground Truth (engineer-only; collapsed)
        with st.expander("Ground Truth (CheXpert valid.csv) — engineer only", expanded=False):
            if valid_df is None:
                st.error("valid.csv not found, cannot show ground truth.")
            else:
                gt_pos, gt_unc, rel_path = find_gt_for_uploaded(uploaded, valid_df, valid_index)

                if rel_path is None:
                    st.info("No ground truth available for this upload (not matched to CheXpert valid set).")
                    st.caption("Tip: This works automatically only for images that exist in datasets/valid/.")
                else:
                    st.caption(f"Matched file: `{rel_path}`")

                    st.markdown("### GT positives")
                    st.write(gt_pos if gt_pos else ["(none)"])

                    if gt_unc:
                        st.caption("Uncertain (-1): " + ", ".join(gt_unc))

                    st.markdown("### GT overlap summary")
                    render_gt_overlap(out, gt_pos)

        # 3) ML / Engineer details (collapsed)
        with st.expander("ML / Engineer view (technical details) — engineer only", expanded=False):
            st.write(f"FINAL: {out['final']}")
            st.write(f"Mode: {out.get('mode', '')}")
            st.write("Primary:", out["primary"])
            st.write("Key findings shown:", out["findings"])
            st.write("All findings (sorted):")
            for d in out["details"]:
                flag = "YES" if d["is_present"] else "NO"
                st.write(
                    f"- {d['label']}: score={d['score']:.3f} | present={flag} | reliability={d['reliability']}"
                )

# =========================
# TAB 2: CheXpert valid (random pick with GT)
# =========================
with tab_valid:
    st.caption("Pick a random image from CheXpert valid. Ground Truth is shown automatically.")

    valid_df = load_valid_df()
    if valid_df is None:
        st.error(f"Missing valid.csv at: {VALID_CSV_PATH}")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            subset = st.selectbox("Subset", ["Any", "Only No Finding", "Only Abnormal (No Finding != 1)"])
        with col2:
            filter_label = st.selectbox("Optional: only GT-positive for label", ["(none)"] + LABELS)

        if st.button("Pick random CheXpert VALID image"):
            df = valid_df

            if subset == "Only No Finding":
                df = df[df["No Finding"] == 1.0]
            elif subset == "Only Abnormal (No Finding != 1)":
                df = df[df["No Finding"] != 1.0]

            if filter_label != "(none)" and filter_label in df.columns:
                df = df[df[filter_label] == 1.0]

            if len(df) == 0:
                st.error("No rows matched your filter.")
            else:
                row = df.sample(1).iloc[0]
                rel_path = row["Path_norm"]
                img_path = os.path.join(DATASET_DIR, rel_path)

                st.write("Picked:", rel_path)

                if not os.path.isfile(img_path):
                    st.error(f"Missing file on disk: {img_path}")
                else:
                    img = Image.open(img_path)
                    out = predictor.predict(img)
                    gt_pos, gt_unc = gt_from_row(row)

                    st.subheader("Input image (CheXpert valid)")
                    st.image(img, use_container_width=True)

                    gt_col, pred_col = st.columns(2)

                    with pred_col:
                        st.markdown("## Model output")
                        report = build_patient_report(out)

                        with st.expander("Clinician / Patient-friendly report", expanded=True):
                            if report["status"] == "normal":
                                st.success("IMPRESSION: No significant abnormality detected (AI draft)")
                            else:
                                st.warning("IMPRESSION: Abnormal study (AI draft)")
                            st.markdown(report["markdown"])

                        with st.expander("ML / Engineer view (technical details)", expanded=False):
                            st.write("FINAL:", out["final"])
                            st.write("Primary:", out["primary"])
                            st.write("Shown findings:", out["findings"])
                            st.write("Devices present:", out["devices_present"])
                            note = out.get("primary_note", "")
                            if note:
                                st.caption(note)

                    with gt_col:
                        st.markdown("## Ground Truth (valid.csv)")
                        st.markdown("### GT positives")
                        st.write(gt_pos if gt_pos else ["(none)"])

                        if gt_unc:
                            st.caption("Uncertain (-1): " + ", ".join(gt_unc))

                        st.markdown("### GT overlap summary")
                        render_gt_overlap(out, gt_pos)
