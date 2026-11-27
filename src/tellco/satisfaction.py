"""Task 4: scores, regression, segmentation."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from .config import ENG, EXP, RANDOM_STATE


def compute_scores(users, exp, km_eng, eng_scaler, km_exp, exp_scaler, worst_exp_cluster):
    common = users.index.intersection(exp.index)
    Xe = eng_scaler.transform(users.loc[common, ENG])
    Xx = exp_scaler.transform(exp.loc[common, EXP])
    less_engaged = int(np.argmin(km_eng.cluster_centers_.sum(axis=1)))
    s = pd.DataFrame(index=common)
    s["Engagement Distance"] = np.linalg.norm(Xe - km_eng.cluster_centers_[less_engaged], axis=1)
    s["Experience Distance"] = np.linalg.norm(Xx - km_exp.cluster_centers_[worst_exp_cluster], axis=1)
    s[["Engagement Score", "Experience Score"]] = MinMaxScaler().fit_transform(
        s[["Engagement Distance", "Experience Distance"]])
    s["Satisfaction Score"] = s[["Engagement Score", "Experience Score"]].mean(axis=1)
    return s


def top_satisfied(scores: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return scores.nlargest(n, "Satisfaction Score")[["Engagement Score", "Experience Score", "Satisfaction Score"]]


def build_models():
    models = {"Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
              "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=3,
                                                     n_jobs=-1, random_state=RANDOM_STATE)}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.08, max_depth=6, subsample=0.9,
                                         colsample_bytree=0.9, n_jobs=-1, random_state=RANDOM_STATE)
    except ImportError:
        pass
    return models


def train_models(features: pd.DataFrame, target: pd.Series, test_size=0.2):
    Xtr, Xte, ytr, yte = train_test_split(features, target, test_size=test_size, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, mdl in build_models().items():
        if name == "XGBoost":
            mdl.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
        else:
            mdl.fit(Xtr, ytr)
        p = mdl.predict(Xte)
        fitted[name] = mdl
        rows.append({"Model": name, "MAE": mean_absolute_error(yte, p),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, p))), "R2": r2_score(yte, p)})
    res = pd.DataFrame(rows).set_index("Model")
    return res, fitted, (Xte, yte)


def segment_users(scores: pd.DataFrame, k: int = 2):
    S = scores[["Engagement Score", "Experience Score"]]
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(S)
    scores = scores.copy()
    scores["Cluster"] = km.labels_
    lo = scores.groupby("Cluster")["Satisfaction Score"].mean().idxmin()
    scores["Segment"] = np.where(scores["Cluster"] == lo, "Less satisfied", "More satisfied")
    return scores, km


def segment_summary(scores: pd.DataFrame) -> pd.DataFrame:
    seg = scores.groupby("Segment").agg(users=("Satisfaction Score", "size"),
                                        engagement=("Engagement Score", "mean"),
                                        experience=("Experience Score", "mean"),
                                        satisfaction=("Satisfaction Score", "mean"))
    seg["share %"] = seg["users"] / seg["users"].sum() * 100
    return seg
