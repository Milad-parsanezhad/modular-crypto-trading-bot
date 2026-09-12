from __future__ import annotations

"""Compute-optimized wrapper for the frozen v0.42 characterization.

Scientific protocol is unchanged.  This wrapper memoizes two deterministic,
seed-independent design expansions used repeatedly by the v0.41 competing-risk
engine:

1) the discrete-time hazard training table for a given fold; and
2) the event-level design matrix for a given DataFrame object.

The cached arrays are byte-for-byte outputs of the original functions.  No
feature, label, threshold, seed, fold, venue, symbol, cost, gate, or model
hyperparameter is changed.
"""

import importlib.util
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import pandas as pd

import research_bot.event_competing_risk_v41 as cr


_ORIGINAL_HAZARD = cr.build_hazard_training_table_v41
_ORIGINAL_EVENT = cr.event_design_matrix_v41
_HAZARD_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, tuple]] = OrderedDict()
_EVENT_CACHE: OrderedDict[tuple, tuple[pd.DataFrame, object]] = OrderedDict()
_MAX_HAZARD_CACHE = 2
_MAX_EVENT_CACHE = 24


def _policy_key(policy) -> tuple:
    p = policy or cr.CompetingRiskPolicyV41()
    return (
        int(p.max_hold_bars),
        float(p.conformal_alpha),
        int(p.minimum_expert_events),
        int(p.minimum_expert_cause_events),
        int(p.minimum_family_conformal_events),
        float(p.hazard_cap),
        bool(p.require_target_probability_gt_stop),
    )


def _cached_hazard(events: pd.DataFrame, feature_columns: Sequence[str], policy=None):
    features = tuple(feature_columns)
    key = (id(events), len(events), features, _policy_key(policy))
    hit = _HAZARD_CACHE.get(key)
    if hit is not None and hit[0] is events:
        _HAZARD_CACHE.move_to_end(key)
        return hit[1]
    value = _ORIGINAL_HAZARD(events, feature_columns, policy)
    _HAZARD_CACHE[key] = (events, value)
    _HAZARD_CACHE.move_to_end(key)
    while len(_HAZARD_CACHE) > _MAX_HAZARD_CACHE:
        _HAZARD_CACHE.popitem(last=False)
    return value


def _cached_event(events: pd.DataFrame, feature_columns: Sequence[str]):
    features = tuple(feature_columns)
    key = (id(events), len(events), features)
    hit = _EVENT_CACHE.get(key)
    if hit is not None and hit[0] is events:
        _EVENT_CACHE.move_to_end(key)
        return hit[1]
    value = _ORIGINAL_EVENT(events, feature_columns)
    _EVENT_CACHE[key] = (events, value)
    _EVENT_CACHE.move_to_end(key)
    while len(_EVENT_CACHE) > _MAX_EVENT_CACHE:
        _EVENT_CACHE.popitem(last=False)
    return value


# Patch only deterministic representation builders before loading the frozen
# v0.42 runner.  Functions imported later by the vectorized predictor resolve
# these exact cached functions; numerical model fitting remains untouched.
cr.build_hazard_training_table_v41 = _cached_hazard
cr.event_design_matrix_v41 = _cached_event

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts" / "run_v42_breadth_cluster_characterization.py"
spec = importlib.util.spec_from_file_location("v42_frozen_characterization", BASE)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen v0.42 characterization runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


if __name__ == "__main__":
    runner.main()
