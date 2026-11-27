import numpy as np
import pandas as pd
import pytest

from tellco.config import APPS


@pytest.fixture(scope="session")
def raw():
    rng = np.random.RandomState(0)
    n = 600
    d = {"Bearer Id": np.arange(n).astype(float), "IMSI": rng.randint(1, 10**6, n).astype(float),
         "MSISDN": rng.randint(1000, 1000 + 120, n).astype(float), "IMEI": rng.randint(1, 10**6, n).astype(float),
         "Duration_ms": rng.gamma(2, 3e6, n),
         "Handset Manufacturer": rng.choice(["Apple", "Samsung", "Huawei", "Nokia"], n, p=[.4, .3, .2, .1]),
         "Handset Type": rng.choice(["A1", "A2", "S1", "S2", "H1", "N1"], n),
         "Total DL (Bytes)": rng.gamma(2, 3e7, n), "Total UL (Bytes)": rng.gamma(2, 3e6, n),
         "TCP DL Retrans. Vol (Bytes)": rng.gamma(1, 1e6, n), "TCP UL Retrans. Vol (Bytes)": rng.gamma(1, 1e5, n),
         "Avg RTT DL (ms)": rng.gamma(3, 15, n), "Avg RTT UL (ms)": rng.gamma(2, 5, n),
         "Avg Bearer TP DL (kbps)": rng.gamma(1, 2000, n), "Avg Bearer TP UL (kbps)": rng.gamma(1, 300, n)}
    for a in APPS:
        d[f"{a} DL (Bytes)"] = rng.gamma(2, 1e6, n)
        d[f"{a} UL (Bytes)"] = rng.gamma(2, 1e5, n)
    df = pd.DataFrame(d)
    df.loc[rng.choice(n, 30, replace=False), "Avg RTT DL (ms)"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "Handset Type"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "Handset Manufacturer"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "MSISDN"] = np.nan
    df.loc[0, "Total DL (Bytes)"] = 1e12      # obvious outlier
    return df


@pytest.fixture(scope="session")
def clean(raw):
    from tellco import data
    return data.add_totals(data.clean_data(raw))
