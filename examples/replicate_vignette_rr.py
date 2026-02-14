from __future__ import annotations

import numpy as np
import pandas as pd

from panel_local_projection import load_dataset, panel_lp


def _run_manual(
    data: pd.DataFrame,
    y_names: list[str],
    x_name: list[str],
    c_name: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    irf = np.full((len(y_names), len(x_name)), np.nan)
    se = np.full((len(y_names), len(x_name)), np.nan)
    for i, y in enumerate(y_names):
        fit = panel_lp(
            data=data,
            Y_name=y,
            X_name=x_name,
            c_name=c_name,
            method=method,
            te=True,
            cumul=False,
            diff=False,
            g=0,
            twc=False,
            dk=False,
            lagX=0,
            lagY=0,
            H=0,
        )
        irf[i, :] = fit.IRF[0, :]
        se[i, :] = fit.se[0, :]
    return irf, se


def _run_auto(data: pd.DataFrame, y0: str, method: str) -> tuple[np.ndarray, np.ndarray]:
    fit = panel_lp(
        data=data,
        Y_name=y0,
        X_name=["CRISIS"],
        c_name=None,
        method=method,
        te=True,
        cumul=False,
        diff=False,
        g=0,
        twc=False,
        dk=False,
        lagX=4,
        lagY=4,
        H=10,
    )
    return fit.IRF, fit.se


def _to_table(irf: np.ndarray, se: np.ndarray, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "horizon": np.arange(irf.shape[0]),
            f"{prefix}_IRF": irf[:, 0],
            f"{prefix}_se": se[:, 0],
        }
    )


def main() -> None:
    gdp = load_dataset("empirical_RR_f4_lngdp_1980_yx")
    unemp = load_dataset("empirical_RR_f4_lnunemp_1980_yx")

    y_names_gdp = [f"f{i}LNGDP" for i in range(11)]
    c_names_gdp = [
        "l1LNGDP",
        "l2LNGDP",
        "l3LNGDP",
        "l4LNGDP",
        "l1CRISIS",
        "l2CRISIS",
        "l3CRISIS",
        "l4CRISIS",
    ]
    y_names_unemp = [f"f{i}UNEMP" for i in range(11)]
    c_names_unemp = [
        "l1UNEMP",
        "l2UNEMP",
        "l3UNEMP",
        "l4UNEMP",
        "l1CRISIS",
        "l2CRISIS",
        "l3CRISIS",
        "l4CRISIS",
    ]

    fe_manual_gdp, fe_manual_gdp_se = _run_manual(gdp, y_names_gdp, ["CRISIS"], c_names_gdp, "FE")
    spj_manual_gdp, spj_manual_gdp_se = _run_manual(gdp, y_names_gdp, ["CRISIS"], c_names_gdp, "SPJ")
    fe_auto_gdp, fe_auto_gdp_se = _run_auto(gdp, "f0LNGDP", "FE")
    spj_auto_gdp, spj_auto_gdp_se = _run_auto(gdp, "f0LNGDP", "SPJ")

    fe_manual_unemp, fe_manual_unemp_se = _run_manual(unemp, y_names_unemp, ["CRISIS"], c_names_unemp, "FE")
    spj_manual_unemp, spj_manual_unemp_se = _run_manual(unemp, y_names_unemp, ["CRISIS"], c_names_unemp, "SPJ")
    fe_auto_unemp, fe_auto_unemp_se = _run_auto(unemp, "f0UNEMP", "FE")
    spj_auto_unemp, spj_auto_unemp_se = _run_auto(unemp, "f0UNEMP", "SPJ")

    print("GDP (manual FE/SPJ):")
    print(_to_table(fe_manual_gdp, fe_manual_gdp_se, "FE").join(_to_table(spj_manual_gdp, spj_manual_gdp_se, "SPJ").drop(columns="horizon")).round(4))
    print()
    print("GDP (auto FE/SPJ):")
    print(_to_table(fe_auto_gdp, fe_auto_gdp_se, "FE").join(_to_table(spj_auto_gdp, spj_auto_gdp_se, "SPJ").drop(columns="horizon")).round(4))
    print()
    print("UNEMP (manual FE/SPJ):")
    print(_to_table(fe_manual_unemp, fe_manual_unemp_se, "FE").join(_to_table(spj_manual_unemp, spj_manual_unemp_se, "SPJ").drop(columns="horizon")).round(4))
    print()
    print("UNEMP (auto FE/SPJ):")
    print(_to_table(fe_auto_unemp, fe_auto_unemp_se, "FE").join(_to_table(spj_auto_unemp, spj_auto_unemp_se, "SPJ").drop(columns="horizon")).round(4))
    print()

    print("Max abs differences (manual - auto):")
    print(
        {
            "GDP_FE_IRF": float(np.nanmax(np.abs(fe_manual_gdp - fe_auto_gdp))),
            "GDP_SPJ_IRF": float(np.nanmax(np.abs(spj_manual_gdp - spj_auto_gdp))),
            "UNEMP_FE_IRF": float(np.nanmax(np.abs(fe_manual_unemp - fe_auto_unemp))),
            "UNEMP_SPJ_IRF": float(np.nanmax(np.abs(spj_manual_unemp - spj_auto_unemp))),
            "GDP_FE_se": float(np.nanmax(np.abs(fe_manual_gdp_se - fe_auto_gdp_se))),
            "GDP_SPJ_se": float(np.nanmax(np.abs(spj_manual_gdp_se - spj_auto_gdp_se))),
            "UNEMP_FE_se": float(np.nanmax(np.abs(fe_manual_unemp_se - fe_auto_unemp_se))),
            "UNEMP_SPJ_se": float(np.nanmax(np.abs(spj_manual_unemp_se - spj_auto_unemp_se))),
        }
    )


if __name__ == "__main__":
    main()

