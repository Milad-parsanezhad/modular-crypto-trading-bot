from __future__ import annotations

"""Frozen v0.48 temporal/regime non-stationarity diagnostic primitives.

Research-only failure attribution. This module does not change trading policy,
model selection, labels, costs, portfolio logic, or external-data state.
"""

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import binomtest, spearmanr, wasserstein_distance

BOOTSTRAP_SEED_V48 = 314159
BOOTSTRAP_REPS_V48 = 2000
BH_Q_V48 = 0.10


def robust_scale_fit(values: pd.Series, eps: float = 1e-12) -> float | None:
    x = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if x.size < 2:
        return None
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    med = float(np.median(x))
    mad_scale = float(np.median(np.abs(x - med)) * 1.4826)
    candidates = [v for v in (iqr_scale, mad_scale) if np.isfinite(v) and v > eps]
    if not candidates:
        return None
    return max(candidates)


def standardized_wasserstein(a: pd.Series, b: pd.Series, scale: float | None) -> float | None:
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return None
    xa = (
        pd.to_numeric(a, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    xb = (
        pd.to_numeric(b, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if xa.size < 2 or xb.size < 2:
        return None
    return float(wasserstein_distance(xa, xb) / scale)


def chronological_halves(frame: pd.DataFrame, time_col: str = "signal_time") -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    ts = pd.to_datetime(frame[time_col], utc=True, errors="raise")
    unique = sorted(pd.Series(ts.unique()).tolist())
    if len(unique) < 2:
        return frame.iloc[0:0].copy(), frame.copy()
    cut = len(unique) // 2
    early_times = set(unique[:cut])
    mask = ts.map(lambda x: x in early_times)
    return frame.loc[mask].copy(), frame.loc[~mask].copy()


def js_divergence(a: Iterable[object], b: Iterable[object]) -> float | None:
    sa = pd.Series(list(a), dtype="object").dropna().astype(str)
    sb = pd.Series(list(b), dtype="object").dropna().astype(str)
    if sa.empty or sb.empty:
        return None
    cats = sorted(set(sa.unique()) | set(sb.unique()))
    pa = sa.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy(dtype=float)
    pb = sb.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy(dtype=float)
    return float(jensenshannon(pa, pb, base=2.0) ** 2)


def one_sided_sign_p(values: Iterable[float]) -> tuple[int, int, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr) & (arr != 0)]
    n = int(arr.size)
    if n == 0:
        return 0, 0, 1.0
    k = int(np.sum(arr > 0))
    return k, n, float(binomtest(k, n, p=0.5, alternative="greater").pvalue)


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    idx = np.where(finite)[0]
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    tmp = np.empty(m, dtype=float)
    tmp[order] = q
    out[idx] = tmp
    return out


def fold_cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    fold_col: str = "fold",
    reps: int = BOOTSTRAP_REPS_V48,
    seed: int = BOOTSTRAP_SEED_V48,
) -> dict:
    d = frame[[fold_col, x_col, y_col]].copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty or d[fold_col].nunique() < 2:
        return {
            "rho": None,
            "ci90_low": None,
            "ci90_high": None,
            "n": int(len(d)),
            "folds": int(d[fold_col].nunique()),
        }
    rho = float(spearmanr(d[x_col], d[y_col]).statistic)
    folds = np.array(sorted(d[fold_col].unique()))
    rng = np.random.default_rng(seed)
    groups = {f: d.loc[d[fold_col].eq(f)].copy() for f in folds}
    boots: list[float] = []
    for _ in range(reps):
        sampled = rng.choice(folds, size=len(folds), replace=True)
        b = pd.concat([groups[f] for f in sampled], ignore_index=True)
        if b[x_col].nunique() < 2 or b[y_col].nunique() < 2:
            continue
        r = float(spearmanr(b[x_col], b[y_col]).statistic)
        if np.isfinite(r):
            boots.append(r)
    if not boots:
        return {
            "rho": rho,
            "ci90_low": None,
            "ci90_high": None,
            "n": int(len(d)),
            "folds": int(len(folds)),
        }
    low, high = np.quantile(np.asarray(boots), [0.05, 0.95])
    return {
        "rho": rho,
        "ci90_low": float(low),
        "ci90_high": float(high),
        "n": int(len(d)),
        "folds": int(len(folds)),
    }
