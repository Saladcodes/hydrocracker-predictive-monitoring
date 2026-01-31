# FILE: src/ofm_fg_ofm/pipelines/generate_synth.py
"""Generate synthetic hydrocracker raw data.

Defaults are tuned for dashboard demos:
- ~5 months
- 5-minute sampling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ofm_fg_ofm.data.synthetic import make_synthetic_hydrocracker


def _n_rows_from_days(days: float, freq: str) -> int:
    step = pd.to_timedelta(freq)
    if step <= pd.Timedelta(0):
        raise ValueError(f"Invalid freq: {freq}")
    n = int(np.ceil(pd.Timedelta(days=float(days), unit="D") / step))
    return max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic raw CSV for the hydrocracker demo")
    ap.add_argument("--out", required=True, help="output raw csv")

    ap.add_argument("--start", type=str, default="2020-01-01", help="start timestamp")
    ap.add_argument("--freq", type=str, default="5min", help="pandas freq string, e.g. 5min")

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--months", type=float, default=5.0, help="approx 30 days/month")
    ap.add_argument("--days", type=float, default=None, help="override months")
    ap.add_argument("--n-rows", type=int, default=None, help="explicit row count override")
    args = ap.parse_args()

    n_rows = args.n_rows
    if n_rows is None:
        span_days = float(args.days) if args.days is not None else float(args.months) * 30.0
        n_rows = _n_rows_from_days(span_days, args.freq)

    df = make_synthetic_hydrocracker(
        n_rows=int(n_rows),
        freq=str(args.freq),
        start=str(args.start),
        seed=int(args.seed),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote synth raw data to {out} with shape {df.shape}")


if __name__ == "__main__":
    main()
