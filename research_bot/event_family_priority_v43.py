from __future__ import annotations

"""Corrected, frozen event-family priority for v0.43.

Historical v0.41 code documented first-match priority but assigned masks in a
way that allowed later families to overwrite earlier families. Frozen historical
results are not rewritten. v0.43 corrects the documented semantics prospectively
before empirical characterization.
"""

import pandas as pd


V43_EVENT_FAMILY_PRIORITY: tuple[str, ...] = (
    "LIQUIDITY_SWEEP",
    "SMC_FVG_STRUCTURE",
    "BROOKS_FAILED_BREAKOUT",
    "SMC_OB_RETEST",
    "ICHIMOKU_PULLBACK",
    "ICHIMOKU_BREAKOUT",
    "BROOKS_H2L2",
    "ICT_MSS",
    "OTHER_MOTHER_EVENT",
)


def assign_event_family_priority_v43(features: pd.DataFrame) -> pd.Series:
    x = features
    required = {
        "mother_event_v39",
        "ict_recent_sweep_down_v39", "ict_recent_sweep_up_v39",
        "canonical_bull_displacement", "canonical_bear_displacement",
        "brooks_failed_breakdown_v39", "brooks_failed_breakout_v39",
        "smc_bull_ob_retest_v39", "smc_bear_ob_retest_v39",
        "ichimoku_pullback_long_v39", "ichimoku_pullback_short_v39",
        "ichimoku_breakout_up_v39", "ichimoku_breakout_down_v39",
        "brooks_h2_v39", "brooks_l2_v39",
        "ict_mss_up_v39", "ict_mss_down_v39",
    }
    missing = required - set(x.columns)
    if missing:
        raise ValueError(f"missing event-family inputs: {sorted(missing)}")

    family = pd.Series("OTHER_MOTHER_EVENT", index=x.index, dtype="object")
    unassigned = x["mother_event_v39"].eq(1).copy()
    masks: list[tuple[str, pd.Series]] = [
        ("LIQUIDITY_SWEEP", x["ict_recent_sweep_down_v39"].eq(1) | x["ict_recent_sweep_up_v39"].eq(1)),
        ("SMC_FVG_STRUCTURE", x["canonical_bull_displacement"].eq(1) | x["canonical_bear_displacement"].eq(1)),
        ("BROOKS_FAILED_BREAKOUT", x["brooks_failed_breakdown_v39"].eq(1) | x["brooks_failed_breakout_v39"].eq(1)),
        ("SMC_OB_RETEST", x["smc_bull_ob_retest_v39"].eq(1) | x["smc_bear_ob_retest_v39"].eq(1)),
        ("ICHIMOKU_PULLBACK", x["ichimoku_pullback_long_v39"].eq(1) | x["ichimoku_pullback_short_v39"].eq(1)),
        ("ICHIMOKU_BREAKOUT", x["ichimoku_breakout_up_v39"].eq(1) | x["ichimoku_breakout_down_v39"].eq(1)),
        ("BROOKS_H2L2", x["brooks_h2_v39"].eq(1) | x["brooks_l2_v39"].eq(1)),
        ("ICT_MSS", x["ict_mss_up_v39"].eq(1) | x["ict_mss_down_v39"].eq(1)),
    ]
    for name, mask in masks:
        take = unassigned & mask.fillna(False)
        family.loc[take] = name
        unassigned.loc[take] = False
    family.loc[~x["mother_event_v39"].eq(1)] = "OTHER_MOTHER_EVENT"
    return family.astype(str)


def attach_event_family_priority_v43(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["event_family_v41"] = assign_event_family_priority_v43(p)
    lookup = p[["series_id", "timestamp", "event_family_v41"]].rename(columns={"timestamp":"signal_time"})
    out = events.merge(lookup, on=["series_id","signal_time"], how="left", validate="many_to_one")
    out["event_family_v41"] = out["event_family_v41"].fillna("OTHER_MOTHER_EVENT")
    return out
