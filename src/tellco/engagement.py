"""Task 2: engagement clustering."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from .config import ENG, PERSONAS, RANDOM_STATE


def top_n_per_metric(users: pd.DataFrame, n: int = 10) -> dict:
    return {m: users[m].nlargest(n) for m in ENG}


def cluster_engagement(users: pd.DataFrame, k: int = 3):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(users[ENG])
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    eng = users[ENG].copy()
    eng["Cluster"] = km.labels_
    if k == 3:
        order = eng.groupby("Cluster")["Total Data"].mean().sort_values().index.tolist()
        persona = dict(zip(order, PERSONAS))
        eng["Persona"] = eng["Cluster"].map(persona)
    return eng, km, scaler, X


def cluster_summary(eng: pd.DataFrame) -> pd.DataFrame:
    return eng.groupby("Persona")[ENG].agg(["min", "max", "mean", "sum"]).loc[PERSONAS]


def elbow(X, ks=range(1, 11)) -> pd.DataFrame:
    return pd.DataFrame({"k": list(ks),
                         "inertia": [KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X).inertia_
                                     for k in ks]})


def silhouette_by_k(X, ks=range(2, 9), sample=15000):
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    out = []
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
        out.append((k, silhouette_score(X[idx], lab[idx])))
    return pd.DataFrame(out, columns=["k", "silhouette"])
