from __future__ import annotations
import pandas as pd
from .quality import mark_bad_quality, replace_bad_quality

def parse_timestamp(df: pd.DataFrame, col: str = "Timestamp") -> pd.Series:
    # supports your mixed 'dd-mm-yyyy' strings
    return pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce")

def basic_clean(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = parse_timestamp(out, col=timestamp_col)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out

def clean_numeric(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in out.columns if c not in exclude and out[c].dtype != "O"]
    bad = mark_bad_quality(out, cols, frozen_steps=10)
    out = replace_bad_quality(out, cols, bad, method="interpolate")
    return out
