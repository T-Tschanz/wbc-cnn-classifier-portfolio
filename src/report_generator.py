"""
report_generator.py

Generates a polished PDF report for a single Streamlit run of the WBC classifier.

Public entry points:
- build_run_report(df, results, class_labels, labels_df, model_name, normalization, threshold) -> bytes
    Adapter for wbc_app.py. Returns PDF bytes.
- generate_pdf_report(all_results_df, class_labels, settings, labels_df=None) -> bytes
    Core builder. Returns PDF bytes.

Requirements (add to requirements.txt):
    reportlab
    matplotlib
    scikit-learn
"""

from __future__ import annotations

from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak,
)


# --------------------------- Utilities -------------------------------------- #

def _fmt_pct(x: float) -> str:
    try:
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "-"


def _fig_to_png_bytes(fig, dpi: int = 180) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _rl_image_from_png_bytes(png_bytes: bytes, width: float = 6.0 * inch) -> RLImage:
    bio = BytesIO(png_bytes)
    img = RLImage(bio)
    # scale to fixed width (maintain aspect ratio)
    iw, ih = img.drawWidth, img.drawHeight
    scale = width / iw
    img.drawWidth = width
    img.drawHeight = ih * scale
    return img


def _merge_truth_on_filename(
    preds_df: pd.DataFrame, labels_df: Optional[pd.DataFrame]
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Try to merge ground-truth labels onto predictions by 'Filename'.
    Returns (merged_df, truth_col_name_used_or_None)
    """
    if labels_df is None or labels_df.empty:
        return preds_df.copy(), None

    # Try to locate a filename column in labels_df
    fname_candidates = ["Filename", "filename", "file", "path", "image", "Image", "image_path"]
    truth_candidates = [
        "TrueLabel",
        "Label",
        "label",
        "y",
        "target",
        "ground_truth",
        "GroundTruth",
        "truth",
    ]

    ldf = labels_df.copy()
    # Find filename column
    fname_col = next((c for c in fname_candidates if c in ldf.columns), None)
    if not fname_col:
        return preds_df.copy(), None

    # Find truth/label column
    truth_col = next((c for c in truth_candidates if c in ldf.columns), None)
    if not truth_col:
        return preds_df.copy(), None

    # Normalize types for merge
    m = pd.merge(
        preds_df,
        ldf[[fname_col, truth_col]].rename(columns={fname_col: "Filename", truth_col: "TrueLabel"}),
        on="Filename",
        how="inner",
    )
    return m, "TrueLabel"


def _counts_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("Prediction")
        .agg(Count=("Prediction", "size"), MeanConfidence=("Confidence", "mean"))
        .reset_index()
        .sort_values(["Count", "Prediction"], ascending=[False, True])
    )
    out["MeanConfidence"] = out["MeanConfidence"].map(lambda x: f"{x:.3f}")
    out["Percent"] = (out["Count"] / max(1, len(df))).map(_fmt_pct)
    return out


def _make_confidence_hist(df: pd.DataFrame) -> bytes:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.hist(df["Confidence"].astype(float).values, bins=20)
    ax.set_title("Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    return _fig_to_png_bytes(fig)


def _make_confusion_matrix_plot(y_true: List[int], y_pred: List[int], class_labels: List[str]) -> bytes:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_labels))))
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_labels)))
    ax.set_yticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    # write counts
    thresh = cm.max() / 2.0 if cm.size > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _fig_to_png_bytes(fig)


def _classification_report_table(
    y_true: List[int], y_pred: List[int], class_labels: List[str]
) -> Table:
    report = classification_report(
        y_true, y_pred, target_names=class_labels, output_dict=True, zero_division=0
    )

    # Build table rows
    headers = ["Class", "Precision", "Recall", "F1-score", "Support"]
    rows = [headers]
    for cls in class_labels:
        r = report.get(cls, {})
        rows.append(
            [
                cls,
                f"{r.get('precision', 0):.3f}",
                f"{r.get('recall', 0):.3f}",
                f"{r.get('f1-score', 0):.3f}",
                int(r.get("support", 0)),
            ]
        )

    # Add macro/micro/weighted avg
    for agg in ["micro avg", "macro avg", "weighted avg"]:
        r = report.get(agg, {})
        rows.append(
            [
                agg,
                f"{r.get('precision', 0):.3f}",
                f"{r.get('recall', 0):.3f}",
                f"{r.get('f1-score', 0):.3f}",
                int(r.get("support", 0)),
            ]
        )

    tbl = Table(rows, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fcfcfc")]),
            ]
        )
    )
    return tbl


def _simple_table(df: pd.DataFrame, max_rows: int = 20) -> Table:
    use = df.copy()
    if len(use) > max_rows:
        use = use.iloc[:max_rows]

    cols = list(use.columns)
    data = [cols] + [list(map(lambda x: f"{x:.3f}" if isinstance(x, float) else x, row)) for row in use.itertuples(index=False, name=None)]

    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fcfcfc")]),
            ]
        )
    )
    return tbl


# --------------------------- Public: Core Builder --------------------------- #

def generate_pdf_report(
    all_results_df: pd.DataFrame,
    class_labels: List[str],
    settings: Dict,
    labels_df: Optional[pd.DataFrame] = None,
) -> bytes:
    """
    Build a polished PDF report summarizing a Streamlit run.

    Parameters
    ----------
    all_results_df : DataFrame
        Must include columns: "Filename", "Prediction", "Confidence".
        Optional: "Accepted" (bool) to indicate threshold filtering.
    class_labels : list[str]
        The ordered class names used by the model head.
    settings : dict
        {
          "model_name": str,
          "normalization": "0-1"|"mean-std",
          "threshold": float,
          "created_at": datetime (optional),
        }
    labels_df : DataFrame, optional
        Ground-truth mapping. Should include a filename column and a truth/label column.
        The function attempts to auto-detect sensible column names.

    Returns
    -------
    bytes
        The full PDF as bytes.
    """
    if all_results_df is None or all_results_df.empty:
        raise ValueError("No results to report.")

    # Defensive copy
    df = all_results_df.copy()

    # Basic sanity
    if "Filename" not in df.columns or "Prediction" not in df.columns or "Confidence" not in df.columns:
        raise ValueError("Results DataFrame must include Filename, Prediction, Confidence columns.")

    # Coerce types
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    # Summary
    summary_df = _counts_summary(df)

    # Merge ground truth if available
    merged_df, truth_col = _merge_truth_on_filename(df, labels_df)

    # Build y_true/y_pred if we have truth
    has_truth = truth_col is not None
    y_true_idx, y_pred_idx = [], []
    if has_truth:
        # Map labels to indices via class_labels
        label_to_idx = {c: i for i, c in enumerate(class_labels)}
        # Only keep rows whose truth AND prediction are in known label set
        m = merged_df[merged_df["Prediction"].isin(class_labels) & merged_df[truth_col].isin(class_labels)]
        y_pred_idx = m["Prediction"].map(label_to_idx).astype(int).tolist()
        y_true_idx = m[truth_col].map(label_to_idx).astype(int).tolist()

    # Images (plots)
    conf_hist_png = _make_confidence_hist(df)
    conf_hist_img = _rl_image_from_png_bytes(conf_hist_png, width=6.5 * inch)

    cm_img = None
    if has_truth and len(y_true_idx) > 0:
        cm_png = _make_confusion_matrix_plot(y_true_idx, y_pred_idx, class_labels)
        cm_img = _rl_image_from_png_bytes(cm_png, width=6.5 * inch)

    # Build doc
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="WBC Classifier Run Report",
    )

    styles = getSampleStyleSheet()
    H1 = styles["Heading1"]
    H2 = styles["Heading2"]
    H3 = styles["Heading3"]
    Body = styles["BodyText"]
    Small = ParagraphStyle("Small", parent=Body, fontSize=9, leading=11)

    story = []

    # --- Title / Overview ---
    story.append(Paragraph("WBC Classifier — Run Report", H1))
    created_at = settings.get("created_at", datetime.utcnow())
    meta_lines = [
        f"<b>Generated:</b> {created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"<b>Model:</b> {settings.get('model_name', 'N/A')}",
        f"<b>Normalization:</b> {settings.get('normalization', 'N/A')}",
        f"<b>Confidence Threshold:</b> {settings.get('threshold', 'N/A')}",
        f"<b>Total Processed:</b> {len(df)}",
    ]
    story.append(Paragraph("<br/>".join(meta_lines), Body))
    story.append(Spacer(1, 0.2 * inch))

    # --- Summary Table ---
    story.append(Paragraph("Prediction Summary", H2))
    story.append(_simple_table(summary_df[["Prediction", "Count", "Percent", "MeanConfidence"]]))
    story.append(Spacer(1, 0.15 * inch))

    # --- Confidence Histogram ---
    story.append(Paragraph("Confidence Distribution", H2))
    story.append(conf_hist_img)
    story.append(Spacer(1, 0.2 * inch))

    # --- Supervised Metrics ---
    if has_truth and len(y_true_idx) > 0:
        story.append(Paragraph("Supervised Evaluation", H2))
        # Confusion Matrix
        if cm_img:
            story.append(Paragraph("Confusion Matrix", H3))
            story.append(cm_img)
            story.append(Spacer(1, 0.15 * inch))

        # Classification Report
        story.append(Paragraph("Classification Report", H3))
        story.append(_classification_report_table(y_true_idx, y_pred_idx, class_labels))
        story.append(Spacer(1, 0.2 * inch))

    # --- Sample of Predictions ---
    story.append(PageBreak())
    story.append(Paragraph("Sample of Predictions", H2))
    sample_cols = ["Filename", "Prediction", "Confidence"]
    sample_cols = [c for c in sample_cols if c in df.columns]
    story.append(_simple_table(df[sample_cols], max_rows=25))
    story.append(Spacer(1, 0.15 * inch))

    # --- Appendix: Full Table (optional large) ---
    if len(df) > 25:
        story.append(PageBreak())
        story.append(Paragraph("Appendix — Full Results", H2))
        # Break into chunks of ~40 rows to fit pages
        chunk = 40
        for i in range(0, len(df), chunk):
            sub = df.iloc[i : i + chunk][sample_cols]
            story.append(_simple_table(sub, max_rows=chunk))
            story.append(Spacer(1, 0.1 * inch))

    # Build PDF
    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf


# --------------------------- Public: App Adapter ---------------------------- #

def build_run_report(
    df: pd.DataFrame,
    results: list,  # not used yet, reserved for future (e.g., embedded thumbnails)
    class_labels: List[str],
    labels_df: Optional[pd.DataFrame],
    model_name: str,
    normalization: str,
    threshold: float,
) -> bytes:
    """
    Adapter used by wbc_app.py.

    Parameters mirror the app context; converts them into the shape expected by
    generate_pdf_report(...).
    """
    # Ensure minimal columns exist + enforce accepted flag if missing
    use = df.copy()
    for col in ("Filename", "Prediction", "Confidence"):
        if col not in use.columns:
            raise ValueError(f"build_run_report: missing required column '{col}'")

    if "Accepted" not in use.columns:
        use["Accepted"] = True  # app already filtered by threshold

    settings = {
        "model_name": model_name,
        "normalization": normalization,
        "threshold": float(threshold),
        "created_at": datetime.utcnow(),
    }

    return generate_pdf_report(
        all_results_df=use,
        class_labels=class_labels,
        settings=settings,
        labels_df=labels_df,
    )


