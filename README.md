# panel-local-projection (Python)

This repository provides a Python implementation of the R package [`zhentaoshi/panel-local-projection`](https://github.com/zhentaoshi/panel-local-projection), implementing the panel local projection estimator with both Fixed Effects (FE) and Split Panel Jackknife (SPJ) methods.

## What is implemented

- Independent variable (`Y_name`)
- Shock variables (`X_name`)
- Control variables (`c_name`)
- FE estimator (`method="FE"`)
- SPJ estimator (`method="SPJ"`)
- Time effect (`te=True`)
- Two-way clustered standard errors (`twc=True`)
- Driscoll-Kraay style standard errors (`dk=True`)
- Automatic horizon generation (`H`), lag construction (`lagX`, `lagY`)
- Cumulative responses (`cumul=True`) with `diff` (`True`: the input is assumed to be differenced data) and `g` (`g=0`: forward difference; `g=1`: backward difference) options
- Dataset loaders for the six CSV files used in the R package vignette

## Install (local)

```bash
pip install -e .
```

## Quick example

This example demonstrates how the function automatically computes the impulse responses for the FE and SPJ estimators given the input parameters lagX, lagY, and H.

```python
from panel_local_projection import load_dataset, panel_lp

data = load_dataset("empirical_RR_f4_lngdp_1980_yx")

fit_fe = panel_lp(
    data=data,
    Y_name="f0LNGDP",
    X_name=["CRISIS"],
    c_name=None,
    method="FE",
    te=True,
    lagX=4,
    lagY=4,
    H=10,
)

fit_spj = panel_lp(
    data=data,
    Y_name="f0LNGDP",
    X_name=["CRISIS"],
    c_name=None,
    method="SPJ",
    te=True,
    lagX=4,
    lagY=4,
    H=10,
)

print(fit_fe.IRF[:, 0])
print(fit_spj.IRF[:, 0])
```

## Vignette-style replication

Run:

```bash
python examples/replicate_vignette_rr.py
```

The script reproduces the Romer-Romer (2017) workflow from the R vignette in two ways:

- Manual horizon-by-horizon regressions
- Automatic multi-horizon regressions

and reports FE/SPJ impulse responses and standard errors.

