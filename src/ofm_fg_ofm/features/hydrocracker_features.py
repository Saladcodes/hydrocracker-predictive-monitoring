# FILE: src/ofm_fg_ofm/features/hydrocracker_features.py
from __future__ import annotations
import pandas as pd


def add_hydrocracker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Non-leaky feature engineering for the synthetic hydrocracker dataset.

    Works with either:
    - reactor_temp / reactor_pressure / feed_rate / h2_rate / delta_p (current generator)
    - temperature / pressure / flow_rate (legacy naming)
    """
    out = df.copy()

    # ---- Canonicalize a few legacy names (do NOT overwrite if already present) ----
    rename_map = {
        "temperature": "reactor_temp",
        "pressure": "reactor_pressure",
        "flow_rate": "feed_rate",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    # ---- Rolling stats (past-only) ----
    for c, win in [("reactor_temp", 30), ("reactor_temp", 120), ("reactor_pressure", 30), ("reactor_pressure", 120),
                   ("feed_rate", 30), ("feed_rate", 120), ("h2_rate", 30), ("delta_p", 30)]:
        if c in out.columns:
            out[f"{c}_roll_{win}"] = out[c].rolling(win, min_periods=max(5, win // 10)).mean()

    # ---- Lag features (anti-leakage: only past values) ----
    lag_cols = [c for c in ["reactor_temp", "reactor_pressure", "feed_rate", "h2_rate", "delta_p"] if c in out.columns]
    for c in lag_cols:
        for lag in [1, 5, 15]:
            out[f"{c}_lag_{lag}"] = out[c].shift(lag)

    # ---- Simple ratios ----
    if "reactor_temp" in out.columns and "reactor_pressure" in out.columns:
        out["temp_pressure_ratio"] = out["reactor_temp"] / out["reactor_pressure"].clip(lower=1e-6)
    if "feed_rate" in out.columns and "reactor_pressure" in out.columns:
        out["feed_pressure_ratio"] = out["feed_rate"] / out["reactor_pressure"].clip(lower=1e-6)

    return out


def build_hydrocracker_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatible alias."""
    return add_hydrocracker_features(df)
