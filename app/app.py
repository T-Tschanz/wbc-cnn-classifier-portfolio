# wbc_app.py
import streamlit as st
import zipfile
import os
import base64
from io import BytesIO

import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

from src.model_manager import ModelManager
from src.confidence_slider import ConfidenceSlider
from src.image_loader import load_image_from_url
from src.performance_monitor import monitor_memory_usage
from src.preprocessing import preprocess_image
from src.predict import classify_image
from src.visualization import generate_saliency_map, overlay_saliency
from src.evaluation import evaluate_predictions
from src.utils import export_results_to_csv
from src.uncertainty import classify_with_gray_zone

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------

st.set_page_config(page_title="WBC Classifier", page_icon="🧪", layout="wide")

# --- Auth disabled for portfolio demo ---
# (kept in source repo to show extensibility)
username = "Portfolio Demo"

# === end auth ===

# -----------------------------------------------------------------------------
# PDF report builder (nice-looking PDF of the current run)
# We will look for either:
#   - report_generator.build_run_report(...)
#   - report_generator.generate_run_report(...)
#   - report_builder.build_run_report(...)
# -----------------------------------------------------------------------------
HAS_PDF_REPORTS = False
_build_report = None

def _wire_pdf_builder():
    """Try to import a compatible PDF builder and store it in _build_report."""
    global HAS_PDF_REPORTS, _build_report
    try:
        import report_generator as _rg  # prefer report_generator
        _build_report = getattr(_rg, "build_run_report", None) or getattr(_rg, "generate_run_report", None)
        if _build_report:
            HAS_PDF_REPORTS = True
            return
    except Exception:
        pass
    try:
        import report_builder as _rb  # optional alternate
        _build_report = getattr(_rb, "build_run_report", None) or getattr(_rb, "generate_run_report", None)
        if _build_report:
            HAS_PDF_REPORTS = True
            return
    except Exception:
        pass

_wire_pdf_builder()



mm =ModelManager()
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_pdf(path: str) -> str:
    """Return an HTML link that lets users download a PDF that lives in the repo."""
    try:
        with open(str(path), "rb") as f:
            pdf_bytes = f.read()
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        filename = os.path.basename(path)
        return (
            f'<a href="data:application/pdf;base64,{b64}" '
            f'download="{filename}">Download User Guide (PDF)</a>'
        )
    except Exception:
        return "<em>User Guide not found in repository.</em>"

# -----------------------------------------------------------------------------
# Sidebar: About & Settings
# -----------------------------------------------------------------------------

GUIDE_PATH = BASE_DIR / "WBC_Classifier_User_Guide_App.pdf"

st.sidebar.title("About")
st.sidebar.markdown(load_pdf(GUIDE_PATH), unsafe_allow_html=True)

st.sidebar.header("Settings & Model")


# Optional: allow a custom upload (keras/h5/zip)
uploaded_custom = st.sidebar.file_uploader(
    "Or upload your custom model (.keras/.h5 or SavedModel zip)",
    type=["keras", "h5", "zip"],
)
if uploaded_custom:
    uploaded_name = mm.upload_custom_model(uploaded_custom)
    st.sidebar.success(f"Loaded {uploaded_name}")

# Pick a model (built-ins + uploaded)
choices = mm.list_model_names()
default_name = "Ensemble (Soft Voting)"
default_idx = choices.index(default_name) if default_name in choices else 0
model_choice = st.sidebar.selectbox("Choose Model", choices, index=default_idx)

# Load the selected model and the preprocessing mode it expects
model, preproc_mode = mm.get_model_and_mode(model_choice)

# Confidence threshold (unchanged)
confidence_slider = ConfidenceSlider()
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
confidence_slider.adjust_threshold(confidence_threshold)

# Keep the old normalization control for legacy models that expect external scaling.
# Disable it when the model handles preprocessing internally.
normalization = st.sidebar.selectbox(
    "Normalization Method",
    ["0-1", "mean-std"],
    disabled=(preproc_mode in ["model", "mobilenet"]),
)

# If the model expects external scaling, let the user’s choice override.
if preproc_mode == "0-1":
    preproc_mode = normalization

image_url = st.sidebar.text_input("Image URL")

labels_file = st.sidebar.file_uploader("Upload true labels CSV (optional)", type=["csv"])
labels_df = pd.read_csv(labels_file) if labels_file and labels_file.type == "text/csv" else None

# IMPORTANT: class order for the models
CLASS_LABELS = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

# --- Gray-zone settings (global thresholds) ---
if "thresholds" not in st.session_state:
    st.session_state["thresholds"] = {
        "global": {"p1": 0.85, "margin": 0.15, "entropy": 1.20},
        "per_class": {},  # optional future overrides e.g., {"Basophil": {"p1":0.90, ...}}
    }

with st.sidebar.expander("Gray-zone settings", expanded=False):
    g = st.session_state["thresholds"]["global"]
    g["p1"] = st.slider("Min top-1 probability (τ_conf)", 0.50, 0.99, g["p1"], 0.01)
    g["margin"] = st.slider("Min (p1 − p2) margin (τ_margin)", 0.00, 0.50, g["margin"], 0.01)
    g["entropy"] = st.slider("Max entropy (τ_entropy)", 0.50, 2.50, g["entropy"], 0.05)
    st.caption("Predictions failing any rule are sent to **GRAY ZONE** for manual review.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
st.title("WBC Classification Application")
uploaded_file = st.file_uploader("Upload Image or ZIP", type=["jpg", "jpeg", "png", "zip"])

def predict_probs(model, tensor: tf.Tensor) -> np.ndarray:
    """Return a 1D softmax probability vector for one image tensor."""
    preds = model(tensor, training=False).numpy()
    probs = preds[0] if preds.ndim > 1 else preds
    probs = np.asarray(probs).astype("float32")
    # If outputs are not normalized, softmax them.
    s = float(np.sum(probs))
    if not np.isfinite(s) or s < 0.99 or s > 1.01:
        probs = tf.nn.softmax(probs).numpy()
    return probs


def process_image(image: Image.Image, filename: str):
    """
    Preprocess, predict, and compute saliency for a single image.

    Returns:
      result: dict with Prediction, Confidence, Top2, Top2_Prob, Margin, Entropy, Decision
      orig:   PIL.Image.Image (original or resized preview)
      sal:    np.ndarray (saliency map)
    """
    # 1) Preprocess
    arr, orig = preprocess_image(image, width=224, height=224, normalization=preproc_mode)
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)   # shape (1, H, W, C)

    # 2) Predict → probabilities
    probs = predict_probs(model, tensor)  # 1D np.ndarray of length == len(CLASS_LABELS)

    # 3) Gray-zone classification
    rz = classify_with_gray_zone(probs, CLASS_LABELS, st.session_state["thresholds"])

    # 4) Saliency for top-1 (even if GRAY_ZONE)
    try:
        sal = generate_saliency_map(model, tensor, rz["top1_idx"])
    except Exception:
        # Fallback: zero map if saliency fails (keeps UI flowing)
        sal = np.zeros((tensor.shape[1], tensor.shape[2]), dtype="float32")

    # 5) Compose result row
    result = {
        "Filename": filename,
        "Prediction": rz["top1_label"],
        "Confidence": rz["top1_prob"],   # p1
        "Top2": rz["top2_label"],
        "Top2_Prob": rz["top2_prob"],    # p2
        "Margin": rz["margin"],          # p1 - p2
        "Entropy": rz["entropy"],
        "Decision": rz["decision"],      # "CONFIDENT" | "GRAY_ZONE"
    }
    return result, orig, sal



results = []  # list of dicts: Filename, Prediction, Confidence, Original (PIL), Saliency (ndarray)
# Keep a light table for ALL predictions (incl. gray-zone) and a separate media list for previews
rows: list[dict] = []     # Filename, Prediction, Confidence, Top2, Margin, Entropy, Decision
media: list[dict] = []    # Same keys + Original, Saliency (for preview)

# --- URL image path
if image_url:
    try:
        img = load_image_from_url(image_url)
        row, orig, sal = process_image(img, image_url)
        rows.append(row)
        media.append({**row, "Original": orig, "Saliency": sal})

        # Informative banners (does not block saving)
        if row["Decision"] == "GRAY_ZONE":
            st.warning(
                f"**GRAY ZONE** for {row['Filename']} → top: {row['Prediction']} "
                f"(p={row['Confidence']:.3f}), runner-up: {row['Top2']} "
                f"(p={row['Top2_Prob']:.3f}), margin={row['Margin']:.3f}, H={row['Entropy']:.2f}"
            )
        elif row["Confidence"] < confidence_slider.threshold:
            st.warning(
                f"Classification confidence ({row['Confidence']:.2f}) below threshold "
                f"({confidence_slider.threshold:.2f}) for URL image."
            )
    except Exception as e:
        st.error(f"Failed to load or classify image from URL. ({e})")

# --- Local upload path (image or ZIP)
if uploaded_file:
    try:
        uploaded_file.seek(0)
        if zipfile.is_zipfile(uploaded_file):
            with zipfile.ZipFile(uploaded_file, "r") as z:
                for fname in z.namelist():
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        with z.open(fname) as f:
                            image = Image.open(f).convert("RGB")
                            row, orig, sal = process_image(image, fname)
                            rows.append(row)
                            media.append({**row, "Original": orig, "Saliency": sal})

                            if row["Decision"] == "GRAY_ZONE":
                                st.warning(
                                    f"**GRAY ZONE** for {row['Filename']} → top: {row['Prediction']} "
                                    f"(p={row['Confidence']:.3f}), runner-up: {row['Top2']} "
                                    f"(p={row['Top2_Prob']:.3f}), margin={row['Margin']:.3f}, H={row['Entropy']:.2f}"
                                )
                            elif row["Confidence"] < confidence_slider.threshold:
                                st.warning(
                                    f"Classification confidence ({row['Confidence']:.2f}) below threshold "
                                    f"({confidence_slider.threshold:.2f}) for {fname}."
                                )
            uploaded_file.seek(0)
        else:
            # Single image (not a ZIP)
            image = Image.open(uploaded_file).convert("RGB")
            row, orig, sal = process_image(image, uploaded_file.name)
            rows.append(row)
            media.append({**row, "Original": orig, "Saliency": sal})

            if row["Decision"] == "GRAY_ZONE":
                st.warning(
                    f"**GRAY ZONE** for {row['Filename']} → top: {row['Prediction']} "
                    f"(p={row['Confidence']:.3f}), runner-up: {row['Top2']} "
                    f"(p={row['Top2_Prob']:.3f}), margin={row['Margin']:.3f}, H={row['Entropy']:.2f}"
                )
            elif row["Confidence"] < confidence_slider.threshold:
                st.warning(
                    f"Classification confidence ({row['Confidence']:.2f}) below threshold "
                    f"({confidence_slider.threshold:.2f}) for {uploaded_file.name}."
                )
    except zipfile.BadZipFile:
        st.error("The uploaded file appears to be a corrupted ZIP archive.")
    except Image.UnidentifiedImageError:
        st.error("Uploaded file is not a valid image.")
    except Exception as e:
        st.error(f"Unexpected error while reading the upload: {e}")

# --- Display results
if rows:
    df = pd.DataFrame(rows)

    st.subheader("Classification Results")
    cols = ["Filename", "Prediction", "Confidence", "Decision", "Top2", "Top2_Prob", "Margin", "Entropy"]
    cols = [c for c in cols if c in df.columns]
    st.dataframe(df[cols], use_container_width=True)

    # Exports
    export_results_to_csv(df[cols])  # full export (incl. GRAY_ZONE rows)

    gray_df = df[df["Decision"] == "GRAY_ZONE"][cols]
    if not gray_df.empty:
        st.info(f"Gray-zone cases detected: **{len(gray_df)}**")
        gz_csv = gray_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Gray-Zone Only (CSV)",
            data=gz_csv,
            file_name="wbc_gray_zone_cases.csv",
            mime="text/csv",
        )

    # Batch summary (WBC differential) — confident predictions as counts + GRAY_ZONE bin
    if len(df) > 1:
        confident_df = df[df["Decision"] == "CONFIDENT"] if "Decision" in df else df
        summary = confident_df["Prediction"].value_counts().rename_axis("Class").reset_index(name="Count")
        gray_count = int((df["Decision"] == "GRAY_ZONE").sum()) if "Decision" in df else 0
        if gray_count:
            summary = pd.concat([summary, pd.DataFrame([{"Class": "GRAY_ZONE", "Count": gray_count}])], ignore_index=True)
        summary["Percent"] = (summary["Count"] / summary["Count"].sum() * 100).round(2)

        st.subheader("Batch Summary (incl. GRAY_ZONE)")
        st.table(summary)

    # Optional evaluation against provided ground-truth labels
    if labels_df is not None:
        try:
            evaluate_predictions(df, labels_df, CLASS_LABELS, st)
        except Exception as e:
            st.warning(f"Evaluation skipped: {e}")

    # -------- Preview (Original + Saliency) ----------
    st.subheader("Preview")
    view = st.radio("Show", ["All", "Confident only", "Gray-zone only"], horizontal=True)
    if view == "Confident only":
        view_items = [m for m in media if m["Decision"] == "CONFIDENT"]
    elif view == "Gray-zone only":
        view_items = [m for m in media if m["Decision"] == "GRAY_ZONE"]
    else:
        view_items = media

    if view_items:
        idx_sel = st.number_input("Select Image Index", 0, len(view_items) - 1, 0)
        sel = view_items[int(idx_sel)]

        # Original preview
        buf = BytesIO()
        sel["Original"].save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="300">', unsafe_allow_html=True)

        # Saliency overlay
        overlay = overlay_saliency(np.array(sel["Original"]), sel["Saliency"])
        st.image(overlay, caption=f"Saliency Map ({sel['Decision']})", width=300)

    # -------- PDF run report ----------
    st.divider()
    st.subheader("Report")
    if HAS_PDF_REPORTS and _build_report:
        if st.button("Build PDF Report"):
            try:
                pdf_bytes = _build_report(
                    df=df,                         # includes gray-zone metrics
                    results=media,                 # contains Original & Saliency
                    class_labels=CLASS_LABELS,
                    labels_df=labels_df,
                    model_name=(model_choice if not uploaded_custom else "Custom (uploaded)"),
                    normalization=normalization,
                    threshold=confidence_slider.threshold,
                )
                st.download_button(
                    "Download PDF Report",
                    data=pdf_bytes,
                    file_name="WBC_Run_Report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Could not build PDF report: {e}")
    else:
        st.info(
            "PDF report is disabled. To enable: keep `report_generator.py` (or `report_builder.py`) "
            "exporting `build_run_report(...) -> bytes`, and ensure `reportlab` is listed in requirements.txt."
        )

# Memory readout
st.sidebar.write(f"Memory Usage: {monitor_memory_usage():.2f} MB")


