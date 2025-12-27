import pandas as pd
from reporting import generate_batch_report

def test_batch_report_single():
    df = pd.DataFrame({
        "filename": ["a.png"],
        "prediction": ["Neutrophil"],
        "confidence": [0.95]
    })
    report = generate_batch_report(df)
    # report should be DataFrame with expected columns
    expected = {"filename", "predicted", "confidence", "count"}
    assert set(report.columns) >= expected

def test_report_csv_export(tmp_path):
    df = pd.DataFrame({
        "filename": ["a.png", "b.png"],
        "prediction": ["Lymphocyte", "Lymphocyte"],
        "confidence": [0.85, 0.88]
    })
    out = tmp_path / "out.csv"
    generate_batch_report(df, csv_path=str(out))
    assert out.exists()
    loaded = pd.read_csv(out)
    assert "filename" in loaded.columns
