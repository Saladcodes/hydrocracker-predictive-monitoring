# FILE: src/ofm_fg_ofm/pipelines/score_ofm.py
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ofm_fg_ofm.models.mspc_pca import score as score_mspc
from ofm_fg_ofm.rules.logic import ewma, persistence, robust_limits


def _ensure_target_exists(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    for c in [
        "y_actual",
        "fuel_gas_y",
        "y",
        "target",
        "fuel_gas",
        "actual",
        "target_fg_energy_day",
        "fg_fuel_gas",
    ]:
        if c in df.columns:
            return c
    raise KeyError(f"Target '{target}' not found. Columns: {list(df.columns)}")


def _normalize_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a usable datetime 'timestamp' column exists and is sorted."""
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    if "Timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Timestamp"], format="mixed", dayfirst=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return df

    # Always produce a timestamp-like column for downstream dashboards
    df["timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="1min")
    return df


def _get_features_for_model(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Build X DataFrame with exactly the training feature columns.

    - Missing columns are created as NaN (pipeline imputers handle it)
    - Extra columns are ignored
    """
    X = df.copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_names]


def _baseline_mask(df: pd.DataFrame, default_frac: float = 0.3) -> pd.Series:
    """Pick a baseline 'normal' region for limit estimation."""
    if "fault_label" in df.columns and (df["fault_label"] == "normal").any():
        return (df["fault_label"] == "normal")
    baseline_n = max(1000, int(default_frac * len(df)))
    return pd.Series(range(len(df))) < baseline_n


def _safe_div(a: np.ndarray, b: float) -> np.ndarray:
    den = float(b) if b and np.isfinite(b) and abs(b) > 1e-12 else 1.0
    return a / den


def score_dataframe(df: pd.DataFrame, soft_bundle: dict, ofm_bundle: dict) -> pd.DataFrame:
    """Score a dataset in-memory.

    Parameters
    ----------
    df:
        Dataset CSV (already processed) containing process variables + target.
    soft_bundle:
        joblib bundle produced by train_soft_sensor.
    ofm_bundle:
        joblib bundle produced by train_ofm.

    Returns
    -------
    pd.DataFrame
        Scores with residuals, MSPC indices, combined anomaly score, and health + uncertainty band.
    """
    df = _normalize_timestamp(df)

    ss = soft_bundle["model"]

    # Some older bundles may not have feature_names
    if "feature_names" not in soft_bundle or not isinstance(soft_bundle["feature_names"], list) or not soft_bundle["feature_names"]:
        drop = {
            "timestamp",
            "Timestamp",
            "y_actual",
            "y_pred",
            "residual",
            "residual_ewma",
            "residual_dev",
            "residual_alarm",
            "t2",
            "spe",
            "mspc_alarm",
            "root_cause",
            "anomaly_score",
            "health",
        }
        features = [c for c in df.columns if c not in drop]
    else:
        features = soft_bundle["feature_names"]

    target = _ensure_target_exists(df, soft_bundle.get("target", "fuel_gas_y"))

    mspc = ofm_bundle["mspc"]

    # Residual baseline stats (new bundles) or fallback to on-the-fly estimation
    ewma_span = int(ofm_bundle.get("ewma_span", 30))
    res_mu = ofm_bundle.get("residual_mu", None)
    res_sigma = ofm_bundle.get("residual_sigma", None)
    residual_z_limit = ofm_bundle.get("residual_z_limit", None)
    residual_z_dev_limit = ofm_bundle.get("residual_z_dev_limit", None)

    # Backward-compatible key
    residual_dev_limit = float(ofm_bundle.get("residual_dev_limit", ofm_bundle.get("residual_limit", np.nan)))

    # IMPORTANT: pass DataFrame (not numpy) so pipelines handle cat/num correctly
    X_df = _get_features_for_model(df, features)
    y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)

    # Predict
    yhat = ss.predict(X_df)
    residual = y - yhat

    # If residual baseline stats are missing, estimate them from a baseline region
    if res_mu is None or res_sigma is None or (not np.isfinite(res_sigma)) or float(res_sigma) < 1e-9:
        base_idx = _baseline_mask(df)
        res_base = pd.Series(residual)[base_idx.to_numpy()]
        res_mu = float(np.nanmean(res_base))
        res_sigma = float(np.nanstd(res_base))
        if not np.isfinite(res_sigma) or res_sigma < 1e-9:
            res_sigma = 1.0

        res_z_base = (res_base - res_mu) / res_sigma
        res_z_ew_base = ewma(res_z_base, span=ewma_span)
        res_z_dev_base = res_z_base - res_z_ew_base

        q = float(ofm_bundle.get("q", 0.99))
        residual_z_limit = float(robust_limits(res_z_base, q=q))
        residual_z_dev_limit = float(robust_limits(res_z_dev_base, q=q))

        # also set backward-compatible residual_dev_limit if missing
        if not np.isfinite(residual_dev_limit):
            res_ew_raw = ewma(res_base, span=ewma_span)
            residual_dev_limit = float(robust_limits((res_base - res_ew_raw), q=q))

    # Final safety
    res_mu = float(res_mu)
    res_sigma = float(res_sigma) if np.isfinite(res_sigma) and float(res_sigma) > 1e-9 else 1.0
    residual_z_limit = float(residual_z_limit) if residual_z_limit is not None and np.isfinite(residual_z_limit) else 3.0
    residual_z_dev_limit = (
        float(residual_z_dev_limit)
        if residual_z_dev_limit is not None and np.isfinite(residual_z_dev_limit)
        else max(1.0, float(residual_dev_limit))
    )

    # ---------------- Residual monitoring (FIXED) ----------------
    res_s = pd.Series(residual)
    residual_ewma = ewma(res_s, span=ewma_span).to_numpy()
    residual_dev = residual - residual_ewma

    residual_z = (residual - res_mu) / res_sigma
    res_z_s = pd.Series(residual_z)
    residual_z_ewma = ewma(res_z_s, span=ewma_span).to_numpy()
    residual_z_dev = residual_z - residual_z_ewma

    # A dimensionless residual score (ratio-to-limit)
    residual_score = np.maximum(
        np.abs(residual_z) / max(1e-12, residual_z_limit),
        np.abs(residual_z_dev) / max(1e-12, residual_z_dev_limit),
    )

    residual_alarm_mask = (np.abs(residual_z) > residual_z_limit) | (np.abs(residual_z_dev) > residual_z_dev_limit)
    residual_alarm = persistence(pd.Series(residual_alarm_mask), on=5, off=10).to_numpy(dtype=bool)

    # ---------------- MSPC scoring ----------------
    # score_mspc expects numeric matrix with the SAME feature ordering as training.
    mspc_features = ofm_bundle.get("mspc_features", getattr(mspc, "feature_names", []))
    if not isinstance(mspc_features, list) or not mspc_features:
        # fall back: numeric columns from X_df
        mspc_features = X_df.select_dtypes(include=[np.number]).columns.tolist()

    X_mspc = X_df.copy()
    for c in mspc_features:
        if c not in X_mspc.columns:
            X_mspc[c] = np.nan
    X_num = X_mspc[mspc_features].to_numpy(dtype=float)

    m = score_mspc(mspc, X_num)
    t2 = m["t2"]
    spe = m["spe"]

    t2_ratio = _safe_div(t2, float(getattr(mspc, "t2_limit", 0.0)))
    spe_ratio = _safe_div(spe, float(getattr(mspc, "spe_limit", 0.0)))
    mspc_score = np.maximum(t2_ratio, spe_ratio)

    mspc_alarm = persistence(pd.Series(mspc_score > 1.0), on=3, off=10).to_numpy(dtype=bool)

    # ---------------- Combined health / anomaly score ----------------
    anomaly_score = np.maximum(residual_score, mspc_score)
    anom_s = pd.Series(anomaly_score)
    anomaly_ewma = ewma(anom_s, span=ewma_span).to_numpy()

    # Uncertainty proxy (EWMA std). This is NOT a statistical CI; it is an intuitive band.
    anom_sigma = anom_s.ewm(span=max(10, ewma_span * 2), adjust=False).std(bias=False).to_numpy()
    anom_sigma = np.nan_to_num(anom_sigma, nan=0.0)

    health = 1.0 / (1.0 + anomaly_ewma)
    health_lower = 1.0 / (1.0 + (anomaly_ewma + anom_sigma))
    health_upper = 1.0 / (1.0 + np.maximum(anomaly_ewma - anom_sigma, 0.0))

    alarm_any = residual_alarm | mspc_alarm

    out_df = pd.DataFrame(
        {
            "timestamp": df["timestamp"],
            "y_actual": y,
            "y_pred": yhat,

            # residuals (raw + standardized)
            "residual": residual,
            "residual_ewma": residual_ewma,
            "residual_dev": residual_dev,
            "residual_z": residual_z,
            "residual_z_ewma": residual_z_ewma,
            "residual_z_dev": residual_z_dev,
            "residual_score": residual_score,
            "residual_alarm": residual_alarm,

            # MSPC
            "t2": t2,
            "spe": spe,
            "mspc_score": mspc_score,
            "mspc_alarm": mspc_alarm,

            # Combined health
            "anomaly_score": anomaly_score,
            "anomaly_ewma": anomaly_ewma,
            "health": health,
            "health_lower": health_lower,
            "health_upper": health_upper,
            "alarm_any": alarm_any,

            # Diagnostics
            "root_cause": m.get("root_cause", np.array([np.nan] * len(df), dtype=object)),

            # Limits (constant per run, but convenient to have in the CSV for dashboards)
            "t2_limit": float(getattr(mspc, "t2_limit", np.nan)),
            "spe_limit": float(getattr(mspc, "spe_limit", np.nan)),
            "residual_z_limit": float(residual_z_limit),
            "residual_z_dev_limit": float(residual_z_dev_limit),
        }
    )

    if "fault_label" in df.columns:
        out_df["fault_label"] = df["fault_label"]

    return out_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--soft", required=True)
    ap.add_argument("--ofm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    soft = joblib.load(args.soft)
    ofm = joblib.load(args.ofm)

    out_df = score_dataframe(df, soft, ofm)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote scores to {out}")


if __name__ == "__main__":
    main()
