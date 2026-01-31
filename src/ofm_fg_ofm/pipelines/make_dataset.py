# FILE: src/ofm_fg_ofm/pipelines/make_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ofm_fg_ofm.data.synthetic import make_synthetic_hydrocracker
from ofm_fg_ofm.features.hydrocracker_features import add_hydrocracker_features


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _n_rows_from_days(days: float, freq: str) -> int:
    """Convert a day-span + pandas frequency string into number of rows."""
    # Pandas supports "5min", "15min", "1H", etc.
    step = pd.to_timedelta(freq)
    if step <= pd.Timedelta(0):
        raise ValueError(f"Invalid freq: {freq}")
    n = int(np.ceil((pd.Timedelta(days=float(days), unit="D") / step)))
    return max(1, n)


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure a canonical `timestamp` column exists and is datetime."""
    out = df.copy()

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    elif "Timestamp" in out.columns:
        # Support mixed day-first strings
        out["timestamp"] = pd.to_datetime(out["Timestamp"], format="mixed", dayfirst=True, errors="coerce")
    else:
        # As a last resort, create a synthetic timestamp index
        out["timestamp"] = pd.date_range("2020-01-01", periods=len(out), freq="1min")

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return out


def _canonicalize_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Canonical target naming: standardize to `fuel_gas_y`
    if "fuel_gas_y" not in out.columns:
        for c in ["target_fg_energy_day", "fg_fuel_gas", "y_actual", "target", "fuel_gas"]:
            if c in out.columns:
                out["fuel_gas_y"] = out[c]
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a processed dataset for OFM/soft-sensor pipelines. "
            "Optionally generates a synthetic raw CSV if the input doesn't exist."
        )
    )
    ap.add_argument(
        "--in",
        dest="inp",
        required=False,
        default=None,
        help="raw input csv path (required unless --generate-synthetic is used)",
    )
    ap.add_argument("--out", required=True, help="processed dataset csv path")

    ap.add_argument("--generate-synthetic", action="store_true", help="generate synthetic raw if --in is missing")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing raw/processed files")

    # Synthetic generation controls
    ap.add_argument("--start", type=str, default="2020-01-01", help="start timestamp for synthetic generation")
    ap.add_argument(
        "--freq",
        type=str,
        default="5min",
        help="pandas freq like 5min, 15min, 1H (default: 5min)",
    )
    ap.add_argument(
        "--months",
        type=float,
        default=5.0,
        help="synthetic span in months (approx 30 days/month). Used if --days is not set.",
    )
    ap.add_argument("--seed", type=int, default=7, help="random seed for synthetic generation")
    ap.add_argument(
        "--days",
        type=float,
        default=None,
        help="synthetic span in days. Overrides --months if provided.",
    )
    ap.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="explicit synthetic row count. Overrides --days/--months if provided.",
    )

    ap.add_argument(
        "--y-noise",
        type=float,
        default=0.6,
        help="adds realism; std dev noise added to fuel_gas_y (set 0 to disable)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)

    def _default_raw_from_out(out_p: Path) -> Path:
        """
        If output is .../data/processed/*.csv -> raw goes to .../data/raw/synthetic_raw.csv
        Otherwise, place it next to the output as synthetic_raw.csv
        """
        if out_p.parent.name.lower() == "processed":
            return out_p.parent.parent / "raw" / "synthetic_raw.csv"
        return out_p.parent / "synthetic_raw.csv"

    if args.inp:
        raw_path = Path(args.inp)
    else:
        if not args.generate_synthetic:
            ap.error("--in is required unless --generate-synthetic is provided.")
        raw_path = _default_raw_from_out(out_path)

    _ensure_parent(raw_path)
    _ensure_parent(out_path)

    # Decide row count if generating
    n_rows: Optional[int] = args.n_rows
    if n_rows is None:
        span_days = float(args.days) if args.days is not None else float(args.months) * 30.0
        n_rows = _n_rows_from_days(span_days, args.freq)

    # 1) raw
    if raw_path.exists() and not args.overwrite:
        df_raw = pd.read_csv(raw_path)
    else:
        if not args.generate_synthetic:
            raise FileNotFoundError(
                f"Raw file not found (or overwrite requested): {raw_path}. "
                "Re-run with --generate-synthetic (or provide a raw csv)."
            )

        df_raw = make_synthetic_hydrocracker(
            n_rows=int(n_rows),
            freq=str(args.freq),
            start=str(args.start),
            seed=int(args.seed),
        )
        df_raw.to_csv(raw_path, index=False)
        print(f"[make_dataset] wrote raw: {len(df_raw):,} rows -> {raw_path}")

    # 2) processed
    df = _coerce_timestamp(df_raw)

    # Canonicalize target naming
    df = _canonicalize_target(df)

    # Add features (non-leaky)
    df = add_hydrocracker_features(df)

    # Optional: add realism noise to target (mostly for synthetic)
    target = "fuel_gas_y"
    if target in df.columns and args.y_noise and float(args.y_noise) > 0:
        rng = np.random.default_rng(int(args.seed))
        df[target] = pd.to_numeric(df[target], errors="coerce") + rng.normal(0.0, float(args.y_noise), size=len(df))

    # Drop rows created by rolling/lag (and rows where target missing)
    keep_cols = ["timestamp", target]
    df = df.dropna(subset=[c for c in keep_cols if c in df.columns]).dropna().reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"[make_dataset] wrote processed: {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()
