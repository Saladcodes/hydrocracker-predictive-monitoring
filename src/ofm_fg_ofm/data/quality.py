from __future__ import annotations
import pandas as pd
import numpy as np

def mark_bad_quality(df: pd.DataFrame, cols: list[str], frozen_steps: int = 10,
                     low: float | None = None, high: float | None = None) -> pd.DataFrame:
    mask = pd.DataFrame(False, index=df.index, columns=cols)
    if low is not None:
        mask |= df[cols] < low
    if high is not None:
        mask |= df[cols] > high
    mask |= df[cols].isna()

    for c in cols:
        s = df[c]
        eq = s.eq(s.shift(1))
        run = eq.groupby((eq != eq.shift()).cumsum()).cumcount() + 1
        frozen = eq & (run >= frozen_steps)
        mask[c] |= frozen.fillna(False)
    return mask

def replace_bad_quality(df: pd.DataFrame, cols: list[str], bad: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c].astype(float).mask(bad[c], np.nan)
        if method == "interpolate":
            s = s.interpolate(limit_direction="both")
        elif method == "last_good":
            s = s.ffill().bfill()
        elif method == "next_good":
            s = s.bfill().ffill()
        elif method == "median":
            s = s.fillna(s.median())
        else:
            raise ValueError(method)
        out[c] = s
    return out
