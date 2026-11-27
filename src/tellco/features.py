"""Per-user aggregation (Task 1 and Task 3) and a tiny reusable feature store."""
import os
import pandas as pd
from .config import APPS


def user_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.dropna(subset=["MSISDN"]).groupby("MSISDN")
    agg = {"Duration_ms": "sum", "Total DL (Bytes)": "sum", "Total UL (Bytes)": "sum"}
    for a in APPS:
        agg[f"{a} Total"] = "sum"
    out = g.agg(agg)
    out.insert(0, "Sessions", g.size())
    out["Total Data"] = out["Total DL (Bytes)"] + out["Total UL (Bytes)"]
    return out


def experience_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["MSISDN"]).copy()
    d["Avg TCP"] = (d["TCP DL Retrans. Vol (Bytes)"] + d["TCP UL Retrans. Vol (Bytes)"]) / 2
    d["Avg RTT"] = (d["Avg RTT DL (ms)"] + d["Avg RTT UL (ms)"]) / 2
    d["Avg Throughput"] = (d["Avg Bearer TP DL (kbps)"] + d["Avg Bearer TP UL (kbps)"]) / 2
    exp = d.groupby("MSISDN")[["Avg TCP", "Avg RTT", "Avg Throughput"]].mean()
    handset = (d.groupby(["MSISDN", "Handset Type"]).size().reset_index(name="n")
               .sort_values("n", ascending=False).drop_duplicates("MSISDN")
               .set_index("MSISDN")["Handset Type"])
    exp["Handset Type"] = handset
    return exp


def top_handsets(df: pd.DataFrame, n_handsets=10, n_manu=3, n_per_manu=5):
    known = df[(df["Handset Type"] != "undefined") & (df["Handset Manufacturer"] != "undefined")]
    top_h = known["Handset Type"].value_counts().head(n_handsets)
    top_m = known["Handset Manufacturer"].value_counts().head(n_manu)
    per_m = {m: known.loc[known["Handset Manufacturer"] == m, "Handset Type"].value_counts().head(n_per_manu)
             for m in top_m.index}
    return top_h, top_m, per_m


DEFAULT_STORE = "data/processed/feature_store/features.csv"


def save_features(frame: pd.DataFrame, path: str = DEFAULT_STORE) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    frame.to_csv(path)


def load_features(path: str = DEFAULT_STORE) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)
