from __future__ import annotations
import numpy as np
import pandas as pd

def ewma(x: pd.Series, span: int = 30) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()

def persistence(mask: pd.Series, on: int = 5, off: int = 5) -> pd.Series:
    out = pd.Series(False, index=mask.index)
    on_count = 0
    off_count = 0
    state = False
    for i, v in enumerate(mask.to_numpy()):
        if bool(v):
            on_count += 1; off_count = 0
        else:
            off_count += 1; on_count = 0
        if not state and on_count >= on:
            state = True
        if state and off_count >= off:
            state = False
        out.iat[i] = state
    return out

def robust_limits(x: pd.Series, q: float = 0.99) -> float:
    return float(x.abs().quantile(q))
