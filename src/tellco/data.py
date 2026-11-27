"""Loading and cleaning (missing values -> mean/mode, outliers -> mean of non-outliers)."""
import pandas as pd
from .config import APPS, ID_COLS


def load_raw(path: str) -> pd.DataFrame:
    """Read the xDR file (.xlsx or .csv) and normalise the duplicated duration/ID columns."""
    df = pd.read_excel(path) if str(path).endswith((".xlsx", ".xls")) else pd.read_csv(path)
    df = df.rename(columns={"MSISDN/Number": "MSISDN", "Dur. (ms).1": "Duration_ms"})
    return df.drop(columns=["Dur. (ms)"], errors="ignore")


def outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def replace_outliers_with_mean(s: pd.Series, k: float = 1.5) -> pd.Series:
    is_out = outlier_mask(s, k)
    return s.mask(is_out, s[~is_out].mean())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["Handset Manufacturer", "Handset Type"]:
        if c in df:
            df[c] = df[c].fillna("undefined")
    num_cols = [c for c in df.select_dtypes("number").columns if c not in ID_COLS]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].mean()).fillna(0)
        df[c] = replace_outliers_with_mean(df[c])
    for c in df.select_dtypes(exclude="number").columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def add_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for a in APPS:
        df[f"{a} Total"] = df[f"{a} DL (Bytes)"] + df[f"{a} UL (Bytes)"]
    df["Total Data"] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]
    return df
