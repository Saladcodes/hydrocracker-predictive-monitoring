from __future__ import annotations
import pandas as pd
import numpy as np

# Molecular weights (g/mol) used in your notebook
MW = {
    "C1": 16.042,
    "C2": 30.068,
    "C2=": 28.052,
    "C3": 44.094,
    "C3=": 42.078,
    "C4-i": 58.12,
    "C4-n": 58.12,
    "C4=i": 56.104,
    "C4=1": 56.104,
    "C4=2c": 56.104,
    "C4=2t": 56.104,
    "C5-i": 72.146,
    "C5-n": 72.146,
    "C6+": 86.172,
    "H2": 2.016,
}

# Lower heating values used in your notebook (units consistent with your conversion)
LHV = {
    "C1": 33990,
    "C2": 60497,
    "C2=": 55969,
    "C3": 86496,
    "C3=": 81477,
    "C4-i": 112088,
    "C4-n": 112528,
    "C4=i": 106836,
    "C4=1": 107526,
    "C4=2c": 107001,
    "C4=2t": 107001,
    "C5-i": 137014,
    "C5-n": 137384,
    "C6+": 163109,
    "H2": 10229,
}

# Expected raw column names (as in your CSV/notebook)
COMP_COLS = {
    "C1": "S.LIMS.Q:U-31100:02514::C1:F",
    "C2": "S.LIMS.Q:U-31100:02514::C2:F",
    "C2=": "S.LIMS.Q:U-31100:02514::C2=:F",
    "C3": "S.LIMS.Q:U-31100:02514::C3:F",
    "C3=": "S.LIMS.Q:U-31100:02514::C3=:F",
    "C4-i": "S.LIMS.Q:U-31100:02514::C4-i:F",
    "C4-n": "S.LIMS.Q:U-31100:02514::C4-n:F",
    "C4=i": "S.LIMS.Q:U-31100:02514::C4=i:F",
    "C4=1": "S.LIMS.Q:U-31100:02514::C4=1:F",
    "C4=2c": "S.LIMS.Q:U-31100:02514::C4=2c:F",
    "C4=2t": "S.LIMS.Q:U-31100:02514::C4=2t:F",
    "C5-i": "S.LIMS.Q:U-31100:02514::C5-i:F",
    "C5-n": "S.LIMS.Q:U-31100:02514::C5-n:F",
    "C6+": "S.LIMS.Q:U-31100:02514::C6+:F",
    "H2": "S.LIMS.Q:U-31100:02514::H2",
}

def compute_fg_mw(df: pd.DataFrame) -> pd.Series:
    mw = 0.0
    for k, col in COMP_COLS.items():
        mw = mw + MW[k] * df[col].astype(float)
    return mw / 100.0

def compute_fg_lhv(df: pd.DataFrame) -> pd.Series:
    lhv = 0.0
    for k, col in COMP_COLS.items():
        lhv = lhv + LHV[k] * df[col].astype(float)
    return lhv / 100.0

def compute_fg_energy_day(df: pd.DataFrame, fh_header: str, sh_header: str) -> pd.Series:
    """Notebook algebra simplification:
    Your Total Fuel Consumption = sum(F4..F10) == (FH_header + SH_header) * factor,
    where factor = 24*sqrt(11/MW)*LHV/5750000
    """
    fg_mw = compute_fg_mw(df)
    fg_lhv = compute_fg_lhv(df)
    factor = 24.0 * np.sqrt(11.0 / fg_mw) * fg_lhv / 5750000.0
    return (df[fh_header].astype(float) + df[sh_header].astype(float)) * factor

def compute_sec(df: pd.DataFrame, fg_energy_day: pd.Series, fh_feed: str, sh_feed: str) -> pd.Series:
    total_feed = df[fh_feed].astype(float) + df[sh_feed].astype(float)
    return fg_energy_day / total_feed.replace(0, np.nan)
