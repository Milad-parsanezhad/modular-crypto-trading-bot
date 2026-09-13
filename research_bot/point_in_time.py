from __future__ import annotations

import pandas as pd


def _utc_ns(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def point_in_time_asof_join(
    decisions: pd.DataFrame,
    features: pd.DataFrame,
    *,
    decision_time_col: str = "timestamp",
    available_time_col: str = "available_at",
    by: str | list[str] | None = "asset",
    feature_prefix: str | None = None,
    tolerance: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Backward as-of join using *availability time*, never event time.

    This is the canonical join for on-chain, news, fundamental, tokenomics and
    other delayed data.  A row can only be attached when ``available_at`` is
    less than or equal to the trading decision timestamp.  Future backfill is
    intentionally impossible.
    """

    if decision_time_col not in decisions.columns:
        raise ValueError(f"missing decision time column: {decision_time_col}")
    if available_time_col not in features.columns:
        raise ValueError(f"missing availability column: {available_time_col}")

    left = decisions.copy()
    right = features.copy()
    order_col = "__point_in_time_input_order__"
    while order_col in left.columns or order_col in right.columns:
        order_col = f"_{order_col}"
    left[order_col] = range(len(left))
    left[decision_time_col] = _utc_ns(left[decision_time_col])
    right[available_time_col] = _utc_ns(right[available_time_col])
    left = left.dropna(subset=[decision_time_col])
    right = right.dropna(subset=[available_time_col])

    by_cols: list[str] = []
    if isinstance(by, str):
        by_cols = [by]
    elif by is not None:
        by_cols = list(by)
    for col in by_cols:
        if col not in left.columns or col not in right.columns:
            raise ValueError(f"point-in-time by column missing on one side: {col}")

    protected = set(by_cols + [available_time_col])
    rename: dict[str, str] = {}
    if feature_prefix:
        for col in right.columns:
            if col not in protected:
                rename[col] = f"{feature_prefix}{col}"
        right = right.rename(columns=rename)

    # pandas.merge_asof requires the ``on`` key to be globally monotonic even
    # when ``by`` is supplied. Sorting by group first resets timestamps at each
    # group boundary and fails for normal interleaved multi-asset panels.
    sort_left = [decision_time_col] + by_cols
    sort_right = [available_time_col] + by_cols
    left = left.sort_values(sort_left).reset_index(drop=True)
    right = right.sort_values(sort_right).reset_index(drop=True)

    tol = pd.Timedelta(tolerance) if tolerance is not None else None
    merged = pd.merge_asof(
        left,
        right,
        left_on=decision_time_col,
        right_on=available_time_col,
        by=by_cols or None,
        direction="backward",
        allow_exact_matches=True,
        tolerance=tol,
    )
    return merged.sort_values(order_col).drop(columns=[order_col]).reset_index(drop=True)


def assert_no_future_availability(
    df: pd.DataFrame,
    *,
    decision_time_col: str = "timestamp",
    available_time_col: str = "available_at",
) -> None:
    if available_time_col not in df.columns:
        return
    decision = _utc_ns(df[decision_time_col])
    available = _utc_ns(df[available_time_col])
    bad = available.notna() & decision.notna() & (available > decision)
    if bool(bad.any()):
        raise AssertionError(f"future-availability leakage detected in {int(bad.sum())} rows")
