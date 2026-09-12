from __future__ import annotations

"""Timestamp-cluster moving-block perturbations for v0.43.

The asset-specific v0.43 learner trains the same asset across multiple venues.
Rows from CoinEx/OKX/KuCoin that share a signal timestamp are contemporaneously
correlated observations and must not be split as if they were independent rows.

This helper resamples contiguous *timestamp clusters*. Each selected timestamp
is copied with all of its venue rows. The target block size remains roughly the
frozen 64 training events, but cluster integrity always takes precedence over an
exact row count.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimestampClusterBootstrapDiagnosticsV43:
    original_rows: int
    original_timestamps: int
    resampled_rows: int
    selected_blocks: int
    target_block_events: int


def timestamp_cluster_block_resample_v43(
    events: pd.DataFrame,
    *,
    seed: int,
    target_block_events: int = 64,
    timestamp_col: str = "signal_time",
    venue_col: str = "venue",
) -> tuple[pd.DataFrame, TimestampClusterBootstrapDiagnosticsV43]:
    """Moving-block bootstrap preserving all rows within each timestamp cluster.

    Blocks start at a random observed timestamp and advance contiguously through
    observed timestamps until the block contains at least ``target_block_events``
    rows. Whole timestamp clusters are appended; they are never split. Blocks are
    repeated until the resample has at least the original number of rows.

    No labels/outcomes are inspected by this function.
    """
    if timestamp_col not in events.columns:
        raise ValueError(f"{timestamp_col} column required")
    if venue_col not in events.columns:
        raise ValueError(f"{venue_col} column required")
    if int(target_block_events) < 2:
        raise ValueError("target_block_events must be >=2")
    if events.empty:
        empty = events.copy().reset_index(drop=True)
        return empty, TimestampClusterBootstrapDiagnosticsV43(0, 0, 0, 0, int(target_block_events))

    ordered = events.copy()
    ordered[timestamp_col] = pd.to_datetime(ordered[timestamp_col], utc=True, errors="raise")
    ordered = ordered.sort_values([timestamp_col, venue_col], kind="mergesort").reset_index(drop=True)

    timestamps = pd.Index(ordered[timestamp_col].drop_duplicates().tolist())
    groups = {
        ts: ordered.index[ordered[timestamp_col].eq(ts)].to_numpy(dtype=np.int64)
        for ts in timestamps
    }
    n_ts = len(timestamps)
    n_rows = len(ordered)
    rng = np.random.default_rng(int(seed))

    selected_positions: list[np.ndarray] = []
    accumulated_rows = 0
    block_count = 0
    while accumulated_rows < n_rows:
        start = int(rng.integers(0, n_ts))
        block_parts: list[np.ndarray] = []
        block_rows = 0
        j = start
        while block_rows < int(target_block_events):
            ts = timestamps[j]
            pos = groups[ts]
            block_parts.append(pos)
            block_rows += len(pos)
            j += 1
            if j >= n_ts:
                j = 0
            if j == start:
                break
        block = np.concatenate(block_parts) if block_parts else np.empty(0, dtype=np.int64)
        if block.size == 0:
            raise RuntimeError("empty timestamp-cluster block")
        selected_positions.append(block)
        accumulated_rows += int(block.size)
        block_count += 1

    positions = np.concatenate(selected_positions)
    out = ordered.iloc[positions].reset_index(drop=True)

    # Fail closed: every occurrence count for a timestamp must be an integer
    # multiple of its original venue-cluster size. This proves no timestamp
    # cluster was partially sampled.
    original_sizes = ordered.groupby(timestamp_col, sort=False).size().to_dict()
    resampled_sizes = out.groupby(timestamp_col, sort=False).size().to_dict()
    for ts, count in resampled_sizes.items():
        base = int(original_sizes[ts])
        if base <= 0 or int(count) % base != 0:
            raise RuntimeError("timestamp cluster integrity violation")

    diag = TimestampClusterBootstrapDiagnosticsV43(
        original_rows=int(n_rows),
        original_timestamps=int(n_ts),
        resampled_rows=int(len(out)),
        selected_blocks=int(block_count),
        target_block_events=int(target_block_events),
    )
    return out, diag
