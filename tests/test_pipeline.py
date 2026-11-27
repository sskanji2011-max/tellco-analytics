import json
import pandas as pd
from tellco import pipeline


def test_pipeline_end_to_end(tmp_path, raw):
    src = tmp_path / "in.csv"
    raw.rename(columns={"MSISDN": "MSISDN/Number"}).to_csv(src, index=False)
    out = tmp_path / "out"
    summary = pipeline.run(str(src), str(out), track=False)
    assert summary["scored_customers"] > 0
    final = pd.read_csv(out / "final_scores.csv")
    assert list(final.columns) == ["MSISDN", "Engagement Score", "Experience Score", "Satisfaction Score", "Segment"]
    assert (out / "feature_store" / "features.csv").exists()
    assert json.load(open(out / "summary.json"))["best_model"]


def test_pipeline_with_mlflow_tracking(tmp_path, raw, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "in.csv"
    raw.rename(columns={"MSISDN": "MSISDN/Number"}).to_csv(src, index=False)
    pipeline.run(str(src), str(tmp_path / "out"), track=True)
    runs = pd.read_csv(tmp_path / "out" / "mlflow_runs.csv")
    assert len(runs) >= 2 and "metrics.r2" in runs.columns
