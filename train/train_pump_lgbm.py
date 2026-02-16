from __future__ import annotations

import os
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier, LGBMRegressor


DATA_CSV = os.getenv("PUMP_SENSOR_CSV", "data/sensor.csv")
OUT_DIR = os.getenv("OUT_DIR", "artifacts")
WINDOW = int(os.getenv("WINDOW", "30"))

# Rolling features can produce lots of NaNs early; we drop rows that are too sparse
BURNIN_NONNA_FRAC = float(os.getenv("BURNIN_NONNA_FRAC", "0.80"))

os.makedirs(OUT_DIR, exist_ok=True)


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop common index columns
    for col in ["Unnamed: 0", "Unnamed:0", "index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def get_sensor_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("sensor_")]
    if not cols:
        raise ValueError("No sensor_* columns found in CSV.")
    return cols


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    y_fail: abnormal risk = NOT NORMAL (BROKEN or RECOVERING)
    y_ttb_hours: hours to next BROKEN segment start (0 if currently BROKEN)
    """
    status = df["machine_status"].astype(str).str.upper().fillna("UNKNOWN")
    is_broken = status.eq("BROKEN").to_numpy()

    # ✅ classification target (more positives, learnable)
    df["y_fail"] = (~status.eq("NORMAL")).astype(int)

    # broken segment start indices
    starts = []
    for i in range(len(is_broken)):
        if is_broken[i] and (i == 0 or not is_broken[i - 1]):
            starts.append(i)

    # next broken start for each row
    next_start_idx = np.full(len(df), -1, dtype=int)
    ptr = 0
    for i in range(len(df)):
        while ptr < len(starts) and starts[ptr] < i:
            ptr += 1
        next_start_idx[i] = starts[ptr] if ptr < len(starts) else -1

    ttb = np.full(len(df), np.nan, dtype=np.float32)
    for i in range(len(df)):
        if is_broken[i]:
            ttb[i] = 0.0
        else:
            j = next_start_idx[i]
            if j == -1:
                ttb[i] = np.nan
            else:
                dt = (df.loc[j, "timestamp"] - df.loc[i, "timestamp"]).total_seconds()
                ttb[i] = float(dt / 3600.0)

    df["y_ttb_hours"] = ttb
    return df


def rolling_features(df: pd.DataFrame, sens_cols: List[str], window: int) -> pd.DataFrame:
    Xraw = df[sens_cols].apply(pd.to_numeric, errors="coerce")

    roll = Xraw.rolling(window=window, min_periods=max(3, window // 5))

    mean = roll.mean().add_prefix("mean__")
    std = roll.std(ddof=0).add_prefix("std__")
    mn = roll.min().add_prefix("min__")
    mx = roll.max().add_prefix("max__")
    last = Xraw.add_prefix("last__")

    first = Xraw.shift(window - 1)
    slope = ((Xraw - first) / max(1, window - 1)).add_prefix("slope__")

    X = pd.concat([mean, std, mn, mx, last, slope], axis=1)
    return X


def burnin_filter(df: pd.DataFrame, X: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    non_na = X.notna().mean(axis=1)
    keep = non_na >= frac

    df2 = df.loc[keep].reset_index(drop=True)
    X2 = X.loc[keep].reset_index(drop=True)

    print(f"Kept {len(X2)} rows after burn-in filter (>= {frac:.0%} non-NaN features).")
    return df2, X2


def drop_constant_features(X: pd.DataFrame) -> pd.DataFrame:
    nunique = X.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"Dropping {len(const_cols)} constant features.")
        X = X.drop(columns=const_cols)
    return X


def train_models(df: pd.DataFrame, X: pd.DataFrame) -> Tuple[Pipeline, Pipeline, Dict[str, float]]:
    y_fail = df["y_fail"].astype(int).to_numpy()

    # regression rows where y_ttb exists
    reg_mask = df["y_ttb_hours"].notna().to_numpy()
    X_reg = X.loc[reg_mask]
    y_ttb = df.loc[reg_mask, "y_ttb_hours"].astype(float).to_numpy()

    clf = LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=20,
        min_child_weight=1e-3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )

    reg = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=20,
        min_child_weight=1e-3,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        force_col_wise=True,
        verbosity=-1,
    )

    clf_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", clf)])
    reg_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", reg)])

    # ---- Time-series CV (robust metrics)
    tscv = TimeSeriesSplit(n_splits=5)

    aucs, aps = [], []
    auc_skips = 0
    ap_skips = 0

    for tr, te in tscv.split(X):
        clf_pipe.fit(X.iloc[tr], y_fail[tr])
        p = clf_pipe.predict_proba(X.iloc[te])[:, 1]

        y_te = y_fail[te]
        if len(np.unique(y_te)) < 2:
            auc_skips += 1
            ap_skips += 1
            continue

        aucs.append(roc_auc_score(y_te, p))
        aps.append(average_precision_score(y_te, p))


    # ---- Regression CV
    maes = []
    mae_skips = 0
    for tr, te in tscv.split(X_reg):
        # if fold too small, skip
        if len(tr) < 1000 or len(te) < 1000:
            mae_skips += 1
            continue
        reg_pipe.fit(X_reg.iloc[tr], y_ttb[tr])
        pred = reg_pipe.predict(X_reg.iloc[te])
        maes.append(mean_absolute_error(y_ttb[te], pred))

    metrics = {
        "clf_auc_mean": float(np.mean(aucs)) if aucs else None,
        "clf_ap_mean": float(np.mean(aps)) if aps else None,
        "clf_auc_folds_used": int(len(aucs)),
        "clf_auc_folds_skipped": int(auc_skips),
        "reg_mae_hours_mean": float(np.mean(maes)) if maes else None,
        "reg_mae_folds_used": int(len(maes)),
        "reg_mae_folds_skipped": int(mae_skips),
        "n_rows_used": int(len(df)),
        "n_reg_rows_used": int(reg_mask.sum()),
        "window": int(WINDOW),
        "burnin_nonna_frac": float(BURNIN_NONNA_FRAC),
        "n_features": int(X.shape[1]),
        "pos_rate": float(y_fail.mean()),
    }

    # ---- Final fit on full data
    clf_pipe.fit(X, y_fail)
    reg_pipe.fit(X_reg, y_ttb)

    return clf_pipe, reg_pipe, metrics



def main():
    df = load_df(DATA_CSV)
    if "machine_status" not in df.columns:
        raise ValueError("Expected 'machine_status' column in dataset.")

    # Sensor columns
    sens_cols = get_sensor_cols(df)

    # ✅ Drop sensors entirely missing
    all_null = [c for c in sens_cols if df[c].isna().all()]
    if all_null:
        print(f"Dropping all-null sensors: {all_null}")
    sens_cols = [c for c in sens_cols if c not in all_null]

    # Targets
    df = add_targets(df)

    # Features
    X = rolling_features(df, sens_cols, WINDOW)

    # Burn-in filter
    df, X = burnin_filter(df, X, BURNIN_NONNA_FRAC)

    # Drop constant columns
    X = drop_constant_features(X)

    # Train
    clf_pipe, reg_pipe, metrics = train_models(df, X)

    # Save artifacts
    failure_path = os.path.join(OUT_DIR, "pump_failure_lgbm.joblib")
    ttb_path = os.path.join(OUT_DIR, "pump_ttb_lgbm.joblib")
    schema_path = os.path.join(OUT_DIR, "feature_schema.json")
    metrics_path = os.path.join(OUT_DIR, "train_metrics.json")

    joblib.dump(clf_pipe, failure_path)
    joblib.dump(reg_pipe, ttb_path)

    schema = {
        "window": WINDOW,
        "sensor_columns": sens_cols,
        "feature_columns": list(X.columns),
        "targets": {
            "y_fail": "machine_status != NORMAL (abnormal risk)",
            "y_ttb_hours": "hours to next BROKEN segment start (0 if BROKEN)",
        },
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n✅ Saved artifacts:")
    print(" -", failure_path)
    print(" -", ttb_path)
    print(" -", schema_path)
    print(" -", metrics_path)
    print("\n✅ Metrics:", metrics)


if __name__ == "__main__":
    main()
