"""Task 3: experience clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .config import EXP, RANDOM_STATE

GOOD, LOWTP, POOR = "Good experience", "Low throughput (low latency)", "Poor experience (high latency)"


def top_bottom_frequent(exp: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    freq = exp[col].round(2).value_counts().head(n)
    return pd.DataFrame({"top": exp[col].nlargest(n).values, "bottom": exp[col].nsmallest(n).values,
                         "most_frequent": freq.index.values[:n], "count": freq.values[:n]})


def handset_means(exp: pd.DataFrame, col: str, min_users: int = 100) -> pd.DataFrame:
    t = exp.groupby("Handset Type")[col].agg(["mean", "median", "count"])
    return t[(t["count"] >= min_users) & (t.index != "undefined")].sort_values("mean", ascending=False)


def cluster_experience(exp: pd.DataFrame, k: int = 3):
    scaler = StandardScaler()
    X = scaler.fit_transform(exp[EXP])
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    exp = exp.copy()
    exp["Cluster"] = km.labels_
    c = km.cluster_centers_
    badness = c[:, 0] + c[:, 1] - c[:, 2]          # more retrans + more latency - more throughput
    worst, best = int(np.argmax(badness)), int(np.argmin(badness))
    names = {i: LOWTP for i in range(k)}
    names[best], names[worst] = GOOD, POOR
    exp["Experience Group"] = exp["Cluster"].map(names)
    return exp, km, scaler, worst
