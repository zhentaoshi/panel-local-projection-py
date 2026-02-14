# panel-local-projection (Python)

Python port of the R package [`zhentaoshi/panel-local-projection`](https://github.com/zhentaoshi/panel-local-projection), focused on matching the original `panelLP()` estimator workflow and outputs.

## What is implemented

- FE estimator (`method="FE"`)
- SPJ estimator (`method="SPJ"`)
- Individual-clustered standard errors
- Two-way clustered standard errors (`twc=True`)
- Driscoll-Kraay style standard errors (`dk=True`)
- Automatic horizon generation (`H`), lag construction (`lagX`, `lagY`)
- Cumulative responses (`cumul=True`) with `diff` and `g` options
- Dataset loaders for the six CSV files used in the R package vignette

## Install (local)

```bash
pip install -e .
```

## Quick start

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

