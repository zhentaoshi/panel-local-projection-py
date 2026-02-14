from __future__ import annotations

import numpy as np

from panel_local_projection import load_dataset, panel_lp


def _run_manual(data, y_names, x_name, c_name, method):
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


def _run_auto(data, y0, method):
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


def test_rr_manual_vs_auto_gdp_close():
    data = load_dataset("empirical_RR_f4_lngdp_1980_yx")
    y_names = [f"f{i}LNGDP" for i in range(11)]
    c_names = ["l1LNGDP", "l2LNGDP", "l3LNGDP", "l4LNGDP", "l1CRISIS", "l2CRISIS", "l3CRISIS", "l4CRISIS"]

    irf_manual, se_manual = _run_manual(data, y_names, ["CRISIS"], c_names, "FE")
    irf_auto, se_auto = _run_auto(data, "f0LNGDP", "FE")

    np.testing.assert_allclose(irf_manual, irf_auto, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(se_manual, se_auto, rtol=1e-5, atol=1e-6)


def test_rr_manual_vs_auto_unemp_spj_close():
    data = load_dataset("empirical_RR_f4_lnunemp_1980_yx")
    y_names = [f"f{i}UNEMP" for i in range(11)]
    c_names = ["l1UNEMP", "l2UNEMP", "l3UNEMP", "l4UNEMP", "l1CRISIS", "l2CRISIS", "l3CRISIS", "l4CRISIS"]

    irf_manual, se_manual = _run_manual(data, y_names, ["CRISIS"], c_names, "SPJ")
    irf_auto, se_auto = _run_auto(data, "f0UNEMP", "SPJ")

    np.testing.assert_allclose(irf_manual, irf_auto, rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(se_manual, se_auto, rtol=2e-2, atol=2e-3)


def test_cumul_diff_g0_sets_h0_to_zero():
    data = load_dataset("original_CS_f3")
    fit = panel_lp(
        data=data,
        Y_name="GRRT_WB",
        X_name=["CRISIS"],
        c_name=None,
        method="FE",
        te=False,
        cumul=True,
        diff=True,
        g=0,
        twc=False,
        dk=False,
        lagX=4,
        lagY=4,
        H=2,
    )
    assert fit.IRF.shape == (3, 1)
    assert fit.se.shape == (3, 1)
    assert float(fit.IRF[0, 0]) == 0.0
    assert float(fit.se[0, 0]) == 0.0
