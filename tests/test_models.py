import numpy as np
from tellco import engagement, experience, features, satisfaction
from tellco.config import ENG, EXP, PERSONAS


def _prep(clean):
    users = features.user_aggregate(clean)
    exp = features.experience_aggregate(clean)
    return users, exp


def test_top_n_per_metric(clean):
    users, _ = _prep(clean)
    t = engagement.top_n_per_metric(users, 5)
    assert set(t) == set(ENG) and all(len(v) == 5 for v in t.values())


def test_cluster_engagement_personas(clean):
    users, _ = _prep(clean)
    eng, km, scaler, X = engagement.cluster_engagement(users)
    assert set(eng["Persona"]) == set(PERSONAS)
    means = eng.groupby("Persona")["Total Data"].mean()
    assert means["Light Users"] < means["Regular Users"] < means["Heavy Users"]
    s = engagement.cluster_summary(eng)
    assert list(s.index) == PERSONAS


def test_elbow_is_decreasing_and_silhouette_range(clean):
    users, _ = _prep(clean)
    *_, X = engagement.cluster_engagement(users)
    el = engagement.elbow(X, range(1, 6))
    assert (el["inertia"].diff().dropna() <= 1e-9).all()
    sil = engagement.silhouette_by_k(X, range(2, 4), sample=100)
    assert sil["silhouette"].between(-1, 1).all()


def test_experience_helpers(clean):
    _, exp = _prep(clean)
    t = experience.top_bottom_frequent(exp, "Avg RTT")
    assert len(t) == 10 and (t["top"].iloc[0] >= t["bottom"].iloc[0])
    hm = experience.handset_means(exp, "Avg Throughput", min_users=1)
    assert hm["mean"].is_monotonic_decreasing


def test_cluster_experience_labels(clean):
    _, exp = _prep(clean)
    out, km, scaler, worst = experience.cluster_experience(exp)
    assert experience.POOR in set(out["Experience Group"]) and experience.GOOD in set(out["Experience Group"])
    assert 0 <= worst < 3


def test_scores_and_segments(clean):
    users, exp = _prep(clean)
    eng, km_e, sc_e, _ = engagement.cluster_engagement(users)
    exp_c, km_x, sc_x, worst = experience.cluster_experience(exp)
    s = satisfaction.compute_scores(users, exp_c, km_e, sc_e, km_x, sc_x, worst)
    for c in ["Engagement Score", "Experience Score", "Satisfaction Score"]:
        assert s[c].between(0, 1).all()
    assert np.allclose(s["Satisfaction Score"], (s["Engagement Score"] + s["Experience Score"]) / 2)
    top = satisfaction.top_satisfied(s, 10)
    assert len(top) == 10 and top["Satisfaction Score"].is_monotonic_decreasing
    seg, km2 = satisfaction.segment_users(s)
    assert set(seg["Segment"]) == {"Less satisfied", "More satisfied"}
    summ = satisfaction.segment_summary(seg)
    assert summ.loc["More satisfied", "satisfaction"] > summ.loc["Less satisfied", "satisfaction"]
    assert abs(summ["share %"].sum() - 100) < 1e-6


def test_train_models_reports_metrics(clean):
    users, exp = _prep(clean)
    eng, km_e, sc_e, _ = engagement.cluster_engagement(users)
    exp_c, km_x, sc_x, worst = experience.cluster_experience(exp)
    s = satisfaction.compute_scores(users, exp_c, km_e, sc_e, km_x, sc_x, worst)
    import pandas as pd
    X = pd.concat([users.loc[s.index, ENG], exp_c.loc[s.index, EXP]], axis=1)
    res, fitted, (Xte, yte) = satisfaction.train_models(X, s["Satisfaction Score"])
    assert {"MAE", "RMSE", "R2"} <= set(res.columns) and "Random Forest" in fitted
    assert res["R2"].max() > 0.5   # tiny synthetic set: just beats the mean baseline
