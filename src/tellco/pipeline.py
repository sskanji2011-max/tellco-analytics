"""End-to-end run: clean -> features -> clusters -> scores -> models -> exports (+ MLflow tracking)."""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import data, engagement, experience, features, satisfaction
from .config import ENG, EXP, RANDOM_STATE


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def track_runs(fitted, best_name, Xte, yte, final_csv, n_features, tracking_uri="sqlite:///mlflow_tellco.db"):
    """Task 4.7: one MLflow run per model (code version, start/end time, params, metrics, artifacts)."""
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("satisfaction-model")
    for name, mdl in fitted.items():
        t0 = time.time()
        with mlflow.start_run(run_name=name.lower().replace(" ", "_")):
            mlflow.set_tags({"code_version": git_sha(), "source": "tellco.pipeline",
                             "selected": str(name == best_name)})
            mlflow.log_params({"model": name, "test_size": 0.2, "seed": RANDOM_STATE, "n_features": n_features,
                               **{k: v for k, v in mdl.get_params().items()
                                  if isinstance(v, (int, float, str, bool)) and v is not None}})
            p = mdl.predict(Xte)
            mlflow.log_metric("mae", float(mean_absolute_error(yte, p)))
            mlflow.log_metric("rmse", float(np.sqrt(mean_squared_error(yte, p))))
            mlflow.log_metric("r2", float(r2_score(yte, p)))
            if name == "XGBoost":
                for i, v in enumerate(mdl.evals_result().get("validation_0", {}).get("rmse", [])):
                    mlflow.log_metric("eval_rmse", float(v), step=i)
            mlflow.sklearn.log_model(mdl, "model", serialization_format="cloudpickle")
            mlflow.log_artifact(final_csv)
            mlflow.set_tag("duration_s", f"{time.time() - t0:.1f}")
    return mlflow.search_runs(experiment_names=["satisfaction-model"])


def export_dashboard_tables(df, users, exp_c, eng, X_eng, scores, out_dir):
    """Small summary tables consumed by the Streamlit dashboard."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from .config import APPS, APP_COLS, PERSONAS
    d = os.path.join(out_dir, "dashboard")
    os.makedirs(d, exist_ok=True)
    top_h, top_m, per_m = features.top_handsets(df)
    top_h.rename("sessions").to_csv(os.path.join(d, "top_handsets.csv"))
    top_m.rename("sessions").to_csv(os.path.join(d, "top_manufacturers.csv"))
    pd.concat({m: s for m, s in per_m.items()}, names=["manufacturer", "handset"]).rename("sessions") \
        .to_csv(os.path.join(d, "top_handsets_per_manufacturer.csv"))
    users[APP_COLS].sum().rename("bytes").to_csv(os.path.join(d, "app_totals.csv"))
    users[APP_COLS + ["Total Data"]].corr().to_csv(os.path.join(d, "app_correlation.csv"))
    pca = PCA().fit(StandardScaler().fit_transform(users[APP_COLS]))
    pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(APPS))],
                  "explained": pca.explained_variance_ratio_}).to_csv(os.path.join(d, "pca.csv"), index=False)
    u = users.copy()
    u["Decile"] = pd.qcut(u["Duration_ms"].rank(method="first"), 10, labels=range(1, 11)).astype(int)
    u.groupby("Decile").agg(users=("Total Data", "size"), total_data=("Total Data", "sum")) \
        .to_csv(os.path.join(d, "deciles.csv"))
    engagement.elbow(X_eng).to_csv(os.path.join(d, "elbow.csv"), index=False)
    eng.groupby("Persona").agg(users=("Sessions", "size"), sessions=("Sessions", "mean"),
                               duration_h=("Duration_ms", lambda s: s.mean() / 3.6e6),
                               data_GB=("Total Data", lambda s: s.mean() / 1e9),
                               total_TB=("Total Data", lambda s: s.sum() / 1e12)).loc[PERSONAS] \
        .to_csv(os.path.join(d, "engagement_clusters.csv"))
    exp_c.groupby("Experience Group").agg(users=("Avg RTT", "size"), avg_tcp_MB=("Avg TCP", lambda s: s.mean() / 1e6),
                                          avg_rtt=("Avg RTT", "mean"), avg_throughput=("Avg Throughput", "mean")) \
        .to_csv(os.path.join(d, "experience_clusters.csv"))
    ex = exp_c[exp_c["Handset Type"] != "undefined"].groupby("Handset Type").agg(
        users=("Avg RTT", "size"), avg_rtt=("Avg RTT", "mean"), avg_throughput=("Avg Throughput", "mean"),
        avg_tcp_MB=("Avg TCP", lambda s: s.mean() / 1e6))
    ex[ex["users"] >= 100].to_csv(os.path.join(d, "handset_experience.csv"))
    satisfaction.segment_summary(scores).to_csv(os.path.join(d, "segments.csv"))
    satisfaction.top_satisfied(scores).to_csv(os.path.join(d, "top10_satisfied.csv"))


def run(input_path: str, out_dir: str = "data/processed", track: bool = True) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    raw = data.load_raw(input_path)
    df = data.add_totals(data.clean_data(raw))

    users = features.user_aggregate(df)
    exp = features.experience_aggregate(df)

    eng, km_eng, eng_scaler, X_eng = engagement.cluster_engagement(users)
    exp_c, km_exp, exp_scaler, worst = experience.cluster_experience(exp)

    scores = satisfaction.compute_scores(users, exp_c, km_eng, eng_scaler, km_exp, exp_scaler, worst)
    users_c, exp_cc = users.loc[scores.index], exp_c.loc[scores.index]
    X = pd.concat([users_c[ENG], exp_cc[EXP]], axis=1)
    res, fitted, (Xte, yte) = satisfaction.train_models(X, scores["Satisfaction Score"])
    best = res["R2"].idxmax()
    scores, km2 = satisfaction.segment_users(scores)

    final = scores[["Engagement Score", "Experience Score", "Satisfaction Score", "Segment"]].reset_index()
    final["MSISDN"] = final["MSISDN"].astype("int64")
    final_csv = os.path.join(out_dir, "final_scores.csv")
    final.to_csv(final_csv, index=False)

    store = pd.concat([users_c, exp_cc[EXP + ["Handset Type", "Experience Group"]],
                       scores[["Engagement Score", "Experience Score", "Satisfaction Score", "Segment"]]], axis=1)
    store["Persona"] = eng.loc[store.index, "Persona"]
    features.save_features(store, os.path.join(out_dir, "feature_store", "features.csv"))
    res.to_csv(os.path.join(out_dir, "model_comparison.csv"))

    export_dashboard_tables(df, users, exp_c, eng, X_eng, scores, out_dir)

    runs = None
    if track:
        runs = track_runs(fitted, best, Xte, yte, final_csv, X.shape[1])
        runs.to_csv(os.path.join(out_dir, "mlflow_runs.csv"), index=False)

    summary = {"customers": int(len(users)), "scored_customers": int(len(scores)), "best_model": best,
               "metrics": res.round(4).to_dict(orient="index")}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run the TellCo analytics pipeline")
    ap.add_argument("--input", required=True, help="telcom_data.xlsx or .csv")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    a = ap.parse_args()
    print(json.dumps(run(a.input, a.out, track=not a.no_track), indent=2))


if __name__ == "__main__":
    main()
