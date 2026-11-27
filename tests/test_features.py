from tellco import features
from tellco.config import ENG, EXP


def test_user_aggregate(clean):
    u = features.user_aggregate(clean)
    assert u.index.is_unique and u.index.notna().all()
    assert u["Sessions"].sum() == clean["MSISDN"].notna().sum()
    assert (u["Total Data"] == u["Total DL (Bytes)"] + u["Total UL (Bytes)"]).all()
    assert set(ENG) <= set(u.columns)


def test_experience_aggregate(clean):
    e = features.experience_aggregate(clean)
    assert set(EXP + ["Handset Type"]) <= set(e.columns)
    assert e["Handset Type"].notna().all()


def test_top_handsets_sizes_and_exclusion(clean):
    h, m, per = features.top_handsets(clean)
    assert len(m) == 3 and len(per) == 3 and all(len(s) <= 5 for s in per.values())
    assert "undefined" not in h.index and "undefined" not in m.index


def test_feature_store_roundtrip(tmp_path, clean):
    u = features.user_aggregate(clean)
    p = str(tmp_path / "fs" / "f.csv")
    features.save_features(u, p)
    back = features.load_features(p)
    assert back.shape == u.shape
