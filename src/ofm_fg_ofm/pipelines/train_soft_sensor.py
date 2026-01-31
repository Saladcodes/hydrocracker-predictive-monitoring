# src/ofm_fg_ofm/pipelines/train_soft_sensor.py
from __future__ import annotations
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge


CANDIDATE_TARGETS = [
    "fuel_gas_y",            # canonical target in this repo
    "target_fg_energy_day",  # generator output (backward compatible)
    "y_actual",              # common in scores files
    "fuel_gas",
    "target",
]


def pick_target_column(df: pd.DataFrame) -> str:
    for c in CANDIDATE_TARGETS:
        if c in df.columns:
            return c
    raise KeyError(
        f"Target column missing. Tried: {CANDIDATE_TARGETS}. "
        f"Found columns: {list(df.columns)[:30]}..."
    )


def time_order(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def build_preprocessor(X: pd.DataFrame, force_dense: bool = False) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=(not force_dense))),
        ]
    )

    # If we have any categorical features and we need HGB, force dense matrices.
    sparse_threshold = 0.0 if force_dense else 0.3

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=sparse_threshold,
    )


def _cv_rmse(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> float:
    """Forward-chaining CV RMSE (with caps to keep runtime sane)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmses: list[float] = []
    for tr_idx, va_idx in tscv.split(X):
        # cap windows (keeps the tuning practical on big datasets)
        tr_idx = tr_idx[-80000:]
        va_idx = va_idx[:30000]
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[va_idx])
        mse = mean_squared_error(y.iloc[va_idx], pred)
        rmse = float(np.sqrt(mse))
        rmses.append(float(rmse))
    return float(np.mean(rmses))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--model", choices=["hgb", "extra_trees", "ensemble"], default="hgb")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = time_order(df)

    target = pick_target_column(df)
    y = pd.to_numeric(df[target], errors="coerce")

    # Drop rows where target missing
    keep = y.notna()
    df = df.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)

    # Features: drop obvious leakage/outputs
    drop_cols = {
        "timestamp",  # time index should NEVER be a model feature
        "fault_label",  # synthetic labels leak truth; not available in real plants
        "fault_state",  # synthetic labels leak truth; not available in real plants
        target,
        "y_pred",
        "residual",
        "residual_ewma",
        "mspc_alarm",
        "residual_alarm",
        "event_id",
        "alarm_any",
    }
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Train/test split by time (forward split)
    n = len(df)
    n_train = int(args.train_frac * n)
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    # ----------------- Model selection -----------------
    if args.model in ["hgb", "extra_trees"]:
        force_dense = (args.model == "hgb")
        pre = build_preprocessor(X_train, force_dense=force_dense)

        if args.model == "hgb":
            base = HistGradientBoostingRegressor(
                loss="squared_error",
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            )
            param_grid = [
                {"model__max_depth": d, "model__learning_rate": lr, "model__max_iter": it}
                for d in [4, 6, 8]
                for lr in [0.03, 0.05, 0.08]
                for it in [200, 300]
            ]
        else:
            base = ExtraTreesRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=2,
            )
            param_grid = [
                {"model__n_estimators": n_est, "model__min_samples_leaf": msl}
                for n_est in [200, 400]
                for msl in [1, 2, 4]
            ]

        pipe = Pipeline([("pre", pre), ("model", base)])

        best_params: dict | None = None
        best_rmse = float("inf")

        for params in param_grid:
            pipe.set_params(**params)
            rmse = _cv_rmse(pipe, X_train, y_train, n_splits=5)
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = params.copy()

        assert best_params is not None
        pipe.set_params(**best_params)
        pipe.fit(X_train, y_train)

        final_model = pipe
        meta = {"best_params": best_params, "cv_rmse": best_rmse}

    else:
        # Ensemble: average of (boosting + bagging + linear)
        base_models = {
            "hgb": HistGradientBoostingRegressor(
                loss="squared_error",
                max_depth=6,
                learning_rate=0.05,
                max_iter=300,
                random_state=42,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=400,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_leaf=2,
            ),
            "ridge": Ridge(alpha=3.0, random_state=42),
        }

        # One pipeline per base model (each owns its preprocessor)
        base_pipes: dict[str, Pipeline] = {}
        base_cv: dict[str, float] = {}

        for name, model in base_models.items():
            pre = build_preprocessor(X_train, force_dense=True)  # safe default
            pipe = Pipeline([("pre", pre), ("model", model)])
            rmse = _cv_rmse(pipe, X_train, y_train, n_splits=5)
            base_pipes[name] = pipe
            base_cv[name] = rmse

        # weights: inverse RMSE (better model → higher weight)
        weights = []
        estimators = []
        for name in ["hgb", "extra_trees", "ridge"]:
            estimators.append((name, base_pipes[name]))
            weights.append(1.0 / max(1e-9, base_cv[name]))

        ens = VotingRegressor(estimators=estimators, weights=weights)
        ens.fit(X_train, y_train)

        final_model = ens
        meta = {"base_cv_rmse": base_cv, "weights": dict(zip(["hgb", "extra_trees", "ridge"], weights))}

    # ----------------- Evaluation -----------------
    pred_tr = final_model.predict(X_train)
    pred_te = final_model.predict(X_test)

    mae_tr = mean_absolute_error(y_train, pred_tr)
    rmse_tr = float(np.sqrt(mean_squared_error(y_train, pred_tr)))
    mae_te = mean_absolute_error(y_test, pred_te)
    rmse_te = float(np.sqrt(mean_squared_error(y_test, pred_te)))

    print(f"[soft_sensor:{args.model}] train MAE={mae_tr:.4f} RMSE={rmse_tr:.4f}")
    print(f"[soft_sensor:{args.model}]  test MAE={mae_te:.4f} RMSE={rmse_te:.4f}")
    print(f"[soft_sensor:{args.model}] meta={meta}")

    bundle = {
        "model": final_model,
        "target": target,
        "feature_names": list(X.columns),
        "meta": meta,
    }
    joblib.dump(bundle, args.out)
    print(f"Saved soft sensor bundle to {args.out}")


if __name__ == "__main__":
    main()
