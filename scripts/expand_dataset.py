from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=70000)
    ap.add_argument("--timestamp-col", default="timestamp")
    ap.add_argument("--shift-minutes", type=int, default=1)
    ap.add_argument("--noise-frac", type=float, default=0.002)  # 0.2% noise
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], errors="coerce")
    df = df.dropna(subset=[args.timestamp_col]).sort_values(args.timestamp_col).reset_index(drop=True)

    n0 = len(df)
    if n0 == 0:
        raise ValueError("No rows after timestamp parsing.")

    reps = int(np.ceil(args.rows / n0))
    big = pd.concat([df] * reps, ignore_index=True).iloc[: args.rows].copy()

    # shift timestamps forward so it looks like longer history
    big[args.timestamp_col] = big[args.timestamp_col] + pd.to_timedelta(
        np.arange(len(big)) * args.shift_minutes, unit="m"
    )

    # add small noise to numeric columns to avoid “perfect overlap”
    num_cols = big.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        x = pd.to_numeric(big[c], errors="coerce")
        scale = np.nanmedian(np.abs(x)) or 1.0
        noise = np.random.normal(0.0, args.noise_frac * scale, size=len(big))
        big[c] = x + noise

    big.to_csv(args.out, index=False)
    print(f"Wrote {len(big)} rows to {args.out}")


if __name__ == "__main__":
    main()
