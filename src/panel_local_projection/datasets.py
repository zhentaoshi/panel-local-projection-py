from __future__ import annotations

from importlib.resources import files

import pandas as pd


_DATA_FILES = {
    "BVX_annual_auto": "BVX_annual_auto.csv",
    "empirical_CS_f3": "empirical_CS_f3.csv",
    "empirical_MSV_auto": "empirical_MSV_auto.csv",
    "empirical_RR_f4_lngdp_1980_yx": "empirical_RR_f4_lngdp_1980_yx.csv",
    "empirical_RR_f4_lnunemp_1980_yx": "empirical_RR_f4_lnunemp_1980_yx.csv",
    "original_CS_f3": "original_CS_f3.csv",
}


def list_datasets() -> list[str]:
    return sorted(_DATA_FILES)


def load_dataset(name: str) -> pd.DataFrame:
    key = name[:-4] if name.endswith(".csv") else name
    if key not in _DATA_FILES:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    path = files("panel_local_projection").joinpath("data", _DATA_FILES[key])
    return pd.read_csv(path)

