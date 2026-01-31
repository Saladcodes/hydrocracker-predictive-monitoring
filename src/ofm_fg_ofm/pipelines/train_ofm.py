# FILE: src/ofm_fg_ofm/pipelines/train_ofm.py
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from ofm_fg_ofm.models.mspc_pca import fit_mspc
from ofm_fg_ofm.rules.logic import ewma, robust_limits




def _get_features_for_model(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align dataframe to training feature columns."""
    X = df.copy()
    for c in feature_names:
        if c not in X.columns:
            X[c] = np.nan
    return X[feature_names]


def _ensure_target_exists(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    # fallbacks if someone changed dataset columns
    for c in ["y_actual", "fuel_gas_y", "y", "target", "fuel_gas", "actual"]:
        if c in df.columns:
            return c
    raise KeyError(f"Target '{target}' not found. Columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--soft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline-frac", type=float, default=0.3)
    # IMPORTANT: keep n_components < n_features so SPE/Q is non-zero.
    # If you pick n_components == n_features, PCA reconstruction is perfect and SPE becomes ~0.
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--q", type=float, default=0.99)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    soft = joblib.load(args.soft)
    ss = soft["model"]
    features = soft["feature_names"]
    target = _ensure_target_exists(df, soft.get("target", "fuel_gas_y"))

    X_df = _get_features_for_model(df, features)
    y = df[target].to_numpy(dtype=float)

    # IMPORTANT: pass DataFrame so pipeline column selection works
    yhat = ss.predict(X_df)
    residual = pd.Series(y - yhat)

    # baseline selection
    if "fault_label" in df.columns and (df["fault_label"] == "normal").any():
        baseline_idx = (df["fault_label"] == "normal")
    else:
        baseline_n = max(1000, int(args.baseline_frac * len(df)))
        baseline_idx = pd.Series(range(len(df))) < baseline_n

    # ---------------- Residual baselines (for soft-sensor performance monitoring) ----------------
    ewma_span = 30
    res_base = pd.to_numeric(residual[baseline_idx], errors="coerce")
    res_mu = float(np.nanmean(res_base))
    res_sigma = float(np.nanstd(res_base))
    if not np.isfinite(res_sigma) or res_sigma < 1e-9:
        res_sigma = 1.0

    # Standardized residual is the most useful thing to chart (dimensionless, comparable across runs)
    res_z_base = (res_base - res_mu) / res_sigma
    res_z_ew_base = ewma(res_z_base, span=ewma_span)
    res_z_dev_base = res_z_base - res_z_ew_base

    # Limits are robust quantiles on the BASELINE distribution
    residual_z_limit = float(robust_limits(res_z_base, q=args.q))
    residual_z_dev_limit = float(robust_limits(res_z_dev_base, q=args.q))

    # Also keep a backward-compatible "residual_limit" (historically used on residual - EWMA(residual))
    res_ew_raw = ewma(res_base, span=ewma_span)
    residual_dev_limit = float(robust_limits((res_base - res_ew_raw), q=args.q))

    # ---------------- MSPC (PCA) baselines ----------------
    # MSPC should be run on numeric features only.
    X_num_df = X_df.select_dtypes(include=[np.number])
    mspc_features = X_num_df.columns.tolist()
    X_num = X_num_df.to_numpy(dtype=float)
    X_base = X_num[baseline_idx.to_numpy()]

    # Enforce n_components < n_features (otherwise SPE collapses to ~0)
    p = int(X_base.shape[1])
    n_comp = int(min(int(args.n_components), max(1, p - 1)))
    mspc = fit_mspc(X_base, feature_names=mspc_features, n_components=n_comp, q=args.q)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "mspc": mspc,
            # Residual monitoring stats
            "residual_mu": float(res_mu),
            "residual_sigma": float(res_sigma),
            "residual_z_limit": float(residual_z_limit),
            "residual_z_dev_limit": float(residual_z_dev_limit),
            "residual_dev_limit": float(residual_dev_limit),
            "ewma_span": int(ewma_span),

            # Backward compatible key (older score scripts expect this)
            "residual_limit": float(residual_dev_limit),

            "soft_target": target,
            "soft_features": features,
            "mspc_features": mspc_features,
            "q": float(args.q),
        },
        out,
    )

    print(
        "[ofm] "
        f"baseline rows={X_base.shape[0]:,}/{len(df):,}  "
        f"n_comp={mspc.pca.n_components_}/{X_base.shape[1]}  "
        f"T2_lim={mspc.t2_limit:.4f}  SPE_lim={mspc.spe_limit:.4f}  "
        f"res_z_lim={residual_z_limit:.3f}  res_z_dev_lim={residual_z_dev_limit:.3f}"
    )
    print(f"Saved OFM bundle to {out}")


if __name__ == "__main__":
    main()
