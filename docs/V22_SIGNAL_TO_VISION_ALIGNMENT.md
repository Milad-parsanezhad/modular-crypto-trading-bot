# v0.22 Signal-to-Vision Temporal Alignment

For a sample ending at bar `t`, the rendered tensor contains the rolling lookback ending at the close of `t`. Any weak structural target attached to that sample must be computable at or before that same close. The supervised future direction label is computed strictly after `t` and is stored as a target only. Downstream strategy-event joining must use `signal_time` and never an exit timestamp.
