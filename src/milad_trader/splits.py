"""Expanding windows, validation-only selection and a forward-label purge."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    number: int
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


def walk_forward(n, config):
    purge = config.horizon + 1
    for k in range(config.folds):
        train_end = config.train_bars + k*config.test_bars
        val_start = train_end + purge
        val_end = val_start + config.validation_bars
        test_start = val_end + purge
        test_end = test_start + config.test_bars
        if test_end > n:
            raise ValueError(f"Fold {k} needs {test_end} feature rows; received {n}")
        yield Fold(k, (0,train_end), (val_start,val_end), (test_start,test_end))
