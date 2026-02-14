from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelLPResult:
    IRF: np.ndarray
    se: np.ndarray

    def to_frame(self, x_names: Sequence[str] | None = None) -> pd.DataFrame:
        horizons = np.arange(self.IRF.shape[0], dtype=int)
        labels = list(x_names) if x_names is not None else [f"x{idx}" for idx in range(self.IRF.shape[1])]
        blocks: list[pd.DataFrame] = []
        for idx, label in enumerate(labels):
            blocks.append(
                pd.DataFrame(
                    {
                        "horizon": horizons,
                        "variable": label,
                        "irf": self.IRF[:, idx],
                        "se": self.se[:, idx],
                    }
                )
            )
        return pd.concat(blocks, ignore_index=True)


def panel_lp(
    data: pd.DataFrame,
    Y_name: str | Sequence[str],
    X_name: str | Sequence[str],
    c_name: str | Sequence[str] | None = None,
    id_name: str | None = None,
    time_name: str | None = None,
    method: str = "SPJ",
    te: bool = False,
    cumul: bool = False,
    diff: bool = False,
    g: int = 0,
    twc: bool = False,
    dk: bool = False,
    lagX: int | None = None,
    lagY: int | None = None,
    H: int = 0,
) -> PanelLPResult:
    """Panel local projections with FE/SPJ estimators.

    This is a direct Python port of the `panelLP()` function from the R package
    `zhentaoshi/panel-local-projection`.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if H < 0:
        raise ValueError("`H` must be non-negative.")
    if g not in (0, 1):
        raise ValueError("`g` must be either 0 or 1.")

    y_names = _as_list(Y_name, "Y_name")
    if len(y_names) != 1:
        raise ValueError("`Y_name` must contain exactly one dependent variable.")
    x_names = _as_list(X_name, "X_name")
    c_names = _as_list(c_name, "c_name", allow_none=True)

    id_col = id_name or data.columns[0]
    time_col = time_name or data.columns[1]

    lag_x = 0 if lagX is None else int(lagX)
    lag_y = 0 if lagY is None else int(lagY)
    if lag_x < 0 or lag_y < 0:
        raise ValueError("`lagX` and `lagY` must be non-negative.")

    needed_columns = [id_col, time_col, *y_names, *x_names, *c_names]
    missing_columns = [col for col in needed_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in `data`: {missing_columns}")

    prepared = _balance_panel(data.copy(), id_col=id_col, time_col=time_col)

    N = prepared[id_col].nunique()
    T0 = prepared[time_col].nunique()
    px = len(x_names)
    h_seq = np.arange(H + 1, dtype=int)

    irf = np.full((len(h_seq), px), np.nan, dtype=float)
    se = np.full((len(h_seq), px), np.nan, dtype=float)

    y_tn = prepared[y_names].to_numpy(dtype=float)
    x_tn_p = prepared[x_names].to_numpy(dtype=float)
    c_tn_p = (
        prepared[c_names].to_numpy(dtype=float)
        if c_names
        else np.empty((len(prepared), 0), dtype=float)
    )

    ymat_t_n = y_tn.reshape((T0, N), order="F")
    xmat_t_np = (
        np.hstack([x_tn_p[:, [ix]].reshape((T0, N), order="F") for ix in range(px)])
        if px > 0
        else np.empty((T0, 0), dtype=float)
    )

    ylagmat_t_nl = np.empty((T0, 0), dtype=float)
    if lag_y >= 1:
        ylagmat_t_nl = np.hstack([_lag_matrix(ymat_t_n, ilag) for ilag in range(1, lag_y + 1)])

    xlagmat_t_npl = np.empty((T0, 0), dtype=float)
    if lag_x >= 1:
        xlagmat_t_npl = np.hstack([_lag_matrix(xmat_t_np, ilag) for ilag in range(1, lag_x + 1)])

    for ih, h in enumerate(h_seq):
        if not cumul:
            yfmat_t_n = _lead_matrix(ymat_t_n, h)
        else:
            cumul_out = _lead_matrix_cumul(ymat_t_n, h=h, g=g, diff=diff)
            if cumul_out is None:
                irf[ih, :] = 0.0
                se[ih, :] = 0.0
                continue
            yfmat_t_n = cumul_out

        yf_tn = yfmat_t_n.reshape((N * T0, 1), order="F")
        ylag_tn_l = (
            ylagmat_t_nl.reshape((N * T0, ylagmat_t_nl.size // (N * T0)), order="F")
            if ylagmat_t_nl.size
            else np.empty((N * T0, 0), dtype=float)
        )
        xlag_tn_pl = (
            xlagmat_t_npl.reshape((N * T0, xlagmat_t_npl.size // (N * T0)), order="F")
            if xlagmat_t_npl.size
            else np.empty((N * T0, 0), dtype=float)
        )

        reg_parts = [yf_tn, x_tn_p, ylag_tn_l, xlag_tn_pl, c_tn_p]
        yf_x_ylag_xlag_c = np.hstack([part for part in reg_parts if part.shape[1] > 0])

        method_upper = method.upper()
        if method_upper == "FE":
            fe_data = _ols_within_dataprepare(N, T0, yf_x_ylag_xlag_c, te=te)
            dep_var = fe_data[:, 0]
            indep_var = fe_data[:, 1:]
            beta_hat = _ols_no_intercept(dep_var, indep_var)
            res_vec = dep_var - indep_var @ beta_hat
            irf[ih, :] = beta_hat[:px]

            if not dk:
                var_hat = _cluster_var(N, T0, indep_var, res_vec, dd_mat=indep_var, twc=twc)
            else:
                var_hat = _dk_var(N, T0, indep_var, h, res_vec, dd_mat=indep_var)
            se[ih, :] = np.sqrt(np.diag(var_hat))[:px]

        elif method_upper == "SPJ":
            spj_data = _ols_within_dataprepare(N, T0, yf_x_ylag_xlag_c, te=te)
            dep_var = spj_data[:, 0]
            indep_var = spj_data[:, 1:]
            beta_all = _ols_no_intercept(dep_var, indep_var)

            complete = _complete_cases(np.hstack([dep_var[:, None], indep_var]))
            complete_by_id = complete.reshape((T0, N), order="F")
            cut = np.full(N, np.nan, dtype=float)
            for i_n in range(N):
                true_positions = np.where(complete_by_id[:, i_n])[0] + 1
                if true_positions.size:
                    cut[i_n] = float(np.floor(np.median(true_positions)))

            yf_x_a = _build_split_matrix(yf_x_ylag_xlag_c, N=N, cut=cut, use_first=True)
            spj_data_a = _ols_within_dataprepare(N, T0, yf_x_a, te=te)
            dep_var_a = spj_data_a[:, 0]
            indep_var_a = spj_data_a[:, 1:]
            beta_a = _ols_no_intercept(dep_var_a, indep_var_a)

            yf_x_b = _build_split_matrix(yf_x_ylag_xlag_c, N=N, cut=cut, use_first=False)
            spj_data_b = _ols_within_dataprepare(N, T0, yf_x_b, te=te)
            dep_var_b = spj_data_b[:, 0]
            indep_var_b = spj_data_b[:, 1:]
            beta_b = _ols_no_intercept(dep_var_b, indep_var_b)

            beta_hat = 2.0 * beta_all - 0.5 * (beta_a + beta_b)
            irf[ih, :] = beta_hat[:px]

            res_vec = dep_var - indep_var @ beta_hat
            xx_mat = indep_var.reshape((T0, -1), order="F")
            xx_mat_a = indep_var_a.reshape((T0, -1), order="F")
            xx_mat_b = indep_var_b.reshape((T0, -1), order="F")
            xx_mat_sub = _split_var(cut=cut, N=N, var=xx_mat, fhalf=xx_mat_a, shalf=xx_mat_b)
            dd_mat = (2.0 * xx_mat - xx_mat_sub).reshape((N * T0, indep_var.shape[1]), order="F")

            if not dk:
                var_hat = _cluster_var(N, T0, indep_var, res_vec, dd_mat=dd_mat, twc=twc)
            else:
                var_hat = _dk_var(N, T0, indep_var, h, res_vec, dd_mat=dd_mat)
            se[ih, :] = np.sqrt(np.diag(var_hat))[:px]
        else:
            raise ValueError("`method` must be either 'FE' or 'SPJ'.")

    return PanelLPResult(IRF=irf, se=se)


def panelLP(*args, **kwargs) -> PanelLPResult:  # noqa: N802
    """R-compatible alias."""
    return panel_lp(*args, **kwargs)


def _as_list(value: str | Sequence[str] | None, name: str, allow_none: bool = False) -> list[str]:
    if value is None:
        if allow_none:
            return []
        raise ValueError(f"`{name}` cannot be None.")
    if isinstance(value, str):
        return [value]
    values = list(value)
    if not values and not allow_none:
        raise ValueError(f"`{name}` cannot be empty.")
    return values


def _balance_panel(data: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    id_levels = data[id_col].drop_duplicates().tolist()
    id_map = {old: idx + 1 for idx, old in enumerate(id_levels)}
    data[id_col] = data[id_col].map(id_map).astype(int)

    time_levels = sorted(data[time_col].drop_duplicates().tolist())
    time_map = {old: idx + 1 for idx, old in enumerate(time_levels)}
    data[time_col] = data[time_col].map(time_map).astype(int)

    all_times = np.arange(1, len(time_levels) + 1, dtype=int)
    blocks: list[pd.DataFrame] = []
    for idx in range(1, len(id_levels) + 1):
        block = data.loc[data[id_col] == idx].copy()
        present = block[time_col].to_numpy(dtype=int)
        missing = all_times[~np.isin(all_times, present)]
        if missing.size:
            add = pd.DataFrame(np.nan, index=np.arange(missing.size), columns=data.columns)
            add[id_col] = idx
            add[time_col] = missing
            block = pd.concat([block, add], ignore_index=True)
        block = block.sort_values(by=time_col, kind="mergesort")
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True)
    return out[data.columns]


def _lag_matrix(x: np.ndarray, n: int) -> np.ndarray:
    lag_x = np.full_like(x, np.nan, dtype=float)
    if n >= x.shape[0]:
        return lag_x
    lag_x[n:, :] = x[:-n, :]
    return lag_x


def _lead_matrix(x: np.ndarray, n: int) -> np.ndarray:
    lead_x = np.full_like(x, np.nan, dtype=float)
    if n >= x.shape[0]:
        return lead_x
    lead_x[:-n or None, :] = x[n:, :]
    return lead_x


def _lead_matrix_cumul(x: np.ndarray, h: int, g: int = 0, diff: bool = False) -> np.ndarray | None:
    x = np.asarray(x, dtype=float)
    Tn, Nc = x.shape
    out = np.full((Tn, Nc), np.nan, dtype=float)

    if not diff:
        for i in range(Nc):
            col = x[:, i]
            y_lead = np.full(Tn, np.nan, dtype=float)
            if h < Tn:
                y_lead[: Tn - h] = col[h:]

            if g == 0:
                y_base = col
            elif g < Tn:
                y_base = np.full(Tn, np.nan, dtype=float)
                y_base[g:] = col[: Tn - g]
            else:
                y_base = np.full(Tn, np.nan, dtype=float)
            out[:, i] = y_lead - y_base
        return out

    if g not in (0, 1):
        raise ValueError("For `diff=True`, only g in {0, 1} is supported.")

    for i in range(Nc):
        col = x[:, i]
        if g == 0:
            if h == 0:
                return None
            stop = Tn - (h - 1)
            for t in range(stop):
                out[t, i] = np.sum(col[t : t + h])
        else:
            stop = Tn - h
            for t in range(stop):
                out[t, i] = np.sum(col[t : t + h + 1])
    return out


def _ols_within_dataprepare(N: int, T0: int, mat: np.ndarray, te: bool) -> np.ndarray:
    data_dn = _demean(type_i=N, type_want=T0, mat=mat)
    if not te:
        return data_dn
    transformed = _transf(N, T0, data_dn, kind="tnl_ntl")
    demeaned_time = _demean(type_i=T0, type_want=N, mat=transformed)
    return _transf(N, T0, demeaned_time, kind="ntl_tnl")


def _demean(type_i: int, type_want: int, mat: np.ndarray) -> np.ndarray:
    out = mat.copy()
    for i in range(type_i):
        start = i * type_want
        stop = start + type_want
        rows = slice(start, stop)
        block = mat[rows, :]
        complete_rows = ~np.isnan(block).any(axis=1)
        for j in range(mat.shape[1]):
            block_mean = np.mean(block[complete_rows, j]) if np.any(complete_rows) else np.nan
            out[rows, j] = out[rows, j] - block_mean
    return out


def _transf(N: int, T0: int, mat: np.ndarray, kind: str) -> np.ndarray:
    col = N if kind == "tnl_ntl" else T0
    out = np.empty_like(mat, dtype=float)
    for j in range(mat.shape[1]):
        out[:, j] = mat[:, j].reshape((-1, col), order="F").T.reshape((-1,), order="F")
    return out


def _ols_no_intercept(dep: np.ndarray, indep: np.ndarray) -> np.ndarray:
    keep = _complete_cases(np.hstack([dep[:, None], indep]))
    if not np.any(keep):
        raise ValueError("No complete observations available for OLS.")
    beta, *_ = np.linalg.lstsq(indep[keep, :], dep[keep], rcond=None)
    return beta


def _cluster_var(
    N: int,
    T0: int,
    indep_var: np.ndarray,
    res_vec: np.ndarray,
    dd_mat: np.ndarray,
    twc: bool,
) -> np.ndarray:
    keep_nt = ~(
        np.isnan(indep_var).any(axis=1)
        | np.isnan(res_vec)
    )
    smp = int(np.sum(keep_nt))
    k = indep_var.shape[1]

    W_N = np.zeros((k, k), dtype=float)
    res_N = res_vec.reshape((T0, N), order="F")

    for i_n in range(N):
        block = slice(i_n * T0, (i_n + 1) * T0)
        dd_i = dd_mat[block, :]
        keep_i = ~(np.isnan(dd_i).any(axis=1) | np.isnan(res_N[:, i_n]))
        n_keep = int(np.sum(keep_i))
        if n_keep == 0:
            continue
        dd_keep = dd_i[keep_i, :]
        res_keep = res_N[keep_i, i_n]
        if n_keep == 1:
            temp = dd_keep.reshape((-1, 1)) * res_keep[0]
            W_N += temp @ temp.T
        else:
            W_N += dd_keep.T @ np.outer(res_keep, res_keep) @ dd_keep

    if not twc:
        W = W_N
    else:
        dd_mat_nt = np.column_stack(
            [
                dd_mat[:, j].reshape((T0, N), order="F").T.reshape((-1,), order="F")
                for j in range(dd_mat.shape[1])
            ]
        )
        W_T = np.zeros((k, k), dtype=float)
        res_T = res_vec.reshape((T0, N), order="F").T
        for i_t in range(T0):
            block = slice(i_t * N, (i_t + 1) * N)
            dd_t = dd_mat_nt[block, :]
            keep_t = ~(np.isnan(dd_t).any(axis=1) | np.isnan(res_T[:, i_t]))
            n_keep_t = int(np.sum(keep_t))
            if n_keep_t == 0:
                continue
            dd_keep = dd_t[keep_t, :]
            res_keep = res_T[keep_t, i_t]
            if n_keep_t == 1:
                temp_t = dd_keep.reshape((-1, 1)) * res_keep[0]
                W_T += temp_t @ temp_t.T
            else:
                W_T += dd_keep.T @ np.outer(res_keep, res_keep) @ dd_keep

        weighted = dd_mat[keep_nt, :] * res_vec[keep_nt][:, None]
        W_NT = weighted.T @ weighted
        W = W_N + W_T - W_NT

    correction = (N / (N - 1.0)) * ((smp - 1.0) / (smp - k))
    W *= correction

    xx = indep_var[keep_nt, :].T @ indep_var[keep_nt, :]
    xx_inv = _safe_inverse(xx)
    return xx_inv @ W @ xx_inv


def _dk_var(
    N: int,
    T0: int,
    indep_var: np.ndarray,
    h: int,
    res_vec: np.ndarray,
    dd_mat: np.ndarray,
) -> np.ndarray:
    T_h = T0 - h
    m_h = int(np.floor(T_h ** 0.25))

    dd_mat_nt = np.column_stack(
        [
            dd_mat[:, j].reshape((T0, N), order="F").T.reshape((-1,), order="F")
            for j in range(dd_mat.shape[1])
        ]
    )
    g_nt = np.zeros((T0, indep_var.shape[1]), dtype=float)
    res_T = res_vec.reshape((T0, N), order="F").T

    for t in range(T0):
        block = slice(t * N, (t + 1) * N)
        dd_t = dd_mat_nt[block, :]
        keep_t = ~(np.isnan(dd_t).any(axis=1) | np.isnan(res_T[:, t]))
        if np.any(keep_t):
            g_nt[t, :] = res_T[keep_t, t].T @ dd_t[keep_t, :] / np.sqrt(N)

    g_eff = g_nt[:T_h, :]
    S_nt = g_eff.T @ g_eff / T_h
    for j in range(1, m_h + 1):
        w_j = 1.0 - j / (m_h + 1.0)
        G1 = g_eff[: T_h - j, :]
        G2 = g_eff[j:T_h, :]
        delta_j = G2.T @ G1 / T_h
        S_nt += w_j * (delta_j + delta_j.T)

    keep_nt = ~(np.isnan(indep_var).any(axis=1) | np.isnan(res_vec))
    xx = indep_var[keep_nt, :].T @ indep_var[keep_nt, :]
    xx_inv = _safe_inverse(xx)
    dk_mat = xx_inv @ S_nt @ xx_inv
    return dk_mat * N * T_h


def _build_split_matrix(yx: np.ndarray, N: int, cut: np.ndarray, use_first: bool) -> np.ndarray:
    cols: list[np.ndarray] = []
    for j in range(yx.shape[1]):
        var = yx[:, j].reshape((-1, N), order="F")
        if use_first:
            var_half = _split_var(cut=cut, N=N, var=var, fhalf=var, shalf=None)
        else:
            var_half = _split_var(cut=cut, N=N, var=var, fhalf=None, shalf=var)
        cols.append(var_half.reshape((-1, 1), order="F"))
    return np.hstack(cols)


def _split_var(
    cut: np.ndarray,
    N: int,
    var: np.ndarray,
    fhalf: np.ndarray | None = None,
    shalf: np.ndarray | None = None,
) -> np.ndarray:
    out = np.full(var.shape, np.nan, dtype=float)
    for ixx in range(var.shape[1]):
        idx = ixx % N
        c = cut[idx]
        if np.isnan(c):
            continue
        c_int = int(c)
        if fhalf is not None:
            out[:c_int, ixx] = fhalf[:c_int, ixx]
        if shalf is not None:
            out[c_int:, ixx] = shalf[c_int:, ixx]
    return out


def _complete_cases(mat: np.ndarray) -> np.ndarray:
    return ~np.isnan(mat).any(axis=1)


def _safe_inverse(mat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat)
