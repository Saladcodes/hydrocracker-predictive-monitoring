"""End-to-end demo pipeline runner.

This script generates a synthetic hydrocracker dataset for a few months at a
configurable sampling interval, then trains:
- soft sensor model (single + optional ensemble)
- OFM/MSPC model(s)
- score CSVs for the Streamlit dashboard

Why a script?
- The repo no longer ships pre-generated data/models (keeps the zip small).
- This gives you a *single* command to recreate everything.

Example:
    python scripts/run_demo_pipeline.py --months 5 --freq 5min --ensemble

After it finishes:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))

    # Ensure "src/" is importable for subprocess `python -m ofm_fg_ofm...`
    env = os.environ.copy()
    src_dir = str(ROOT / "src")

    existing = env.get("PYTHONPATH", "")
    paths = [p for p in existing.split(os.pathsep) if p] if existing else []
    if src_dir not in paths:
        env["PYTHONPATH"] = src_dir + (os.pathsep + existing if existing else "")

    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=float, default=5.0)
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--freq", type=str, default="5min")
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ensemble", action="store_true", help="also train ensemble soft-sensor + OFM")
    ap.add_argument("--clean", action="store_true", help="remove existing generated artifacts first")
    args = ap.parse_args()

    data_processed = ROOT / "data" / "processed" / "dataset.csv"
    models_dir = ROOT / "models"
    outputs_dir = ROOT / "outputs"

    if args.clean:
        for p in [data_processed, models_dir / "soft_sensor.joblib", models_dir / "ofm.joblib", outputs_dir / "scores_single.csv"]:
            if p.exists():
                p.unlink()
        for p in [models_dir / "soft_sensor_ensemble.joblib", models_dir / "ofm_ensemble.joblib", outputs_dir / "scores_ensemble.csv"]:
            if p.exists():
                p.unlink()

    # 1) Dataset
    cmd = [
        sys.executable,
        "-m",
        "ofm_fg_ofm.pipelines.make_dataset",
        "--out",
        str(data_processed),
        "--generate-synthetic",
        "--freq",
        str(args.freq),
        "--start",
        str(args.start),
        "--seed",
        str(args.seed),
    ]
    if args.days is not None:
        cmd += ["--days", str(args.days)]
    else:
        cmd += ["--months", str(args.months)]

    _run(cmd)

    # 2) Soft sensor
    _run(
        [
            sys.executable,
            "-m",
            "ofm_fg_ofm.pipelines.train_soft_sensor",
            "--data",
            str(data_processed),
            "--out",
            str(models_dir / "soft_sensor.joblib"),
        ]
    )

    if args.ensemble:
        _run(
            [
                sys.executable,
                "-m",
                "ofm_fg_ofm.pipelines.train_soft_sensor",
                "--data",
                str(data_processed),
                "--model",
                "ensemble",
                "--out",
                str(models_dir / "soft_sensor_ensemble.joblib"),
            ]
        )

    # 3) Train OFM (MSPC)
    _run(
        [
            sys.executable,
            "-m",
            "ofm_fg_ofm.pipelines.train_ofm",
            "--data",
            str(data_processed),
            "--soft",
            str(models_dir / "soft_sensor.joblib"),
            "--out",
            str(models_dir / "ofm.joblib"),
        ]
    )

    if args.ensemble:
        _run(
            [
                sys.executable,
                "-m",
                "ofm_fg_ofm.pipelines.train_ofm",
                "--data",
                str(data_processed),
                "--soft",
                str(models_dir / "soft_sensor_ensemble.joblib"),
                "--out",
                str(models_dir / "ofm_ensemble.joblib"),
            ]
        )

    # 4) Score
    _run(
        [
            sys.executable,
            "-m",
            "ofm_fg_ofm.pipelines.score_ofm",
            "--data",
            str(data_processed),
            "--soft",
            str(models_dir / "soft_sensor.joblib"),
            "--ofm",
            str(models_dir / "ofm.joblib"),
            "--out",
            str(outputs_dir / "scores_single.csv"),
        ]
    )


    if args.ensemble:
        _run(
            [
                sys.executable,
                "-m",
                "ofm_fg_ofm.pipelines.score_ofm",
                "--data",
                str(data_processed),
                "--soft",
                str(models_dir / "soft_sensor_ensemble.joblib"),
                "--ofm",
                str(models_dir / "ofm_ensemble.joblib"),
                "--out",
                str(outputs_dir / "scores_ensemble.csv"),
            ]
        )

    print("\nDone. Launch the dashboard:\n  streamlit run dashboard/app.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
