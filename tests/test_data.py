import pandas as pd
from tellco import data
from tellco.config import ID_COLS


def test_outlier_mask_flags_extreme_value():
    s = pd.Series([1, 2, 3, 2, 1, 2, 3, 100.0])
    m = data.outlier_mask(s)
    assert m.iloc[-1] and m.sum() == 1


def test_replace_outliers_uses_mean_of_non_outliers():
    s = pd.Series([1, 2, 3, 2, 1, 2, 3, 100.0])
    out = data.replace_outliers_with_mean(s)
    assert out.iloc[-1] == s.iloc[:-1].mean()
    assert out.iloc[:-1].equals(s.iloc[:-1])


def test_clean_data_fills_missing_and_keeps_ids(raw):
    c = data.clean_data(raw)
    assert c.drop(columns=ID_COLS).isna().sum().sum() == 0
    assert c["MSISDN"].isna().sum() == raw["MSISDN"].isna().sum()   # IDs never imputed
    assert (c["Handset Type"] == "undefined").sum() == raw["Handset Type"].isna().sum()


def test_clean_data_removes_outlier(raw):
    c = data.clean_data(raw)
    assert c["Total DL (Bytes)"].max() < 1e11


def test_clean_data_does_not_mutate_input(raw):
    before = raw.copy()
    data.clean_data(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_add_totals_consistent(clean):
    assert (clean["Total Data"] == clean["Total DL (Bytes)"] + clean["Total UL (Bytes)"]).all()
    assert (clean["Gaming Total"] == clean["Gaming DL (Bytes)"] + clean["Gaming UL (Bytes)"]).all()


def test_load_raw_csv_renames(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"MSISDN/Number": [1], "Dur. (ms)": [5], "Dur. (ms).1": [9]}).to_csv(p, index=False)
    df = data.load_raw(str(p))
    assert list(df.columns) == ["MSISDN", "Duration_ms"] and df["Duration_ms"].iloc[0] == 9
