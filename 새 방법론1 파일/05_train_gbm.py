"""Methodology 05: LightGBM on domain-engineered workload features
(acute:chronic ratios, rest patterns, velocity/spin/extension trend
deltas, season/career cumulative load) instead of the flattened
20 x 33 raw sequence used by scenarios 01-04.

Train/Val/Test split, targets, and resampling policy are unchanged from
the shared pipeline: Train 2016-2021 / Val 2022-2023 / Test 2024-2025,
no resampling applied to Val/Test, class imbalance handled only via
train-time sample weights (event-aware, same scheme as 01_XGBoost_CPU).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score

from modeling_common import DATA, RESULTS, save_predictions_and_metrics, write_run

NON_FEATURE_COLS = ["player_id", "game_date", "split", "target_100d", "regression_days",
                     "event_weight", "evaluation_cohort", "seen_in_train"]
CATEGORICAL = ["role", "p_throws"]


def threshold_for_f1(y, p):
    candidates = np.unique(np.quantile(p, np.linspace(.01, .99, 199)))
    scores = [f1_score(y, p >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def main(seed: int):
    df = pd.read_parquet(DATA / "engineered_features_05.parquet")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    labeled = df["target_100d"].notna()
    train = labeled & df["split"].eq("train")
    val = labeled & df["split"].eq("validation")

    x_train, y_train = df.loc[train, feature_cols], df.loc[train, "target_100d"].astype(int)
    x_val, y_val = df.loc[val, feature_cols], df.loc[val, "target_100d"].astype(int)
    x_all = df[feature_cols]

    event_weight = df.loc[train, "event_weight"].fillna(1).to_numpy(float)
    y_train_np = y_train.to_numpy()
    class_factor = float((y_train_np == 0).sum() / event_weight[y_train_np == 1].sum())
    sample_weight = np.where(y_train_np == 1, event_weight * class_factor, 1.0)

    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=.02, num_leaves=15, max_depth=-1,
        min_child_samples=60, subsample=.8, subsample_freq=1, colsample_bytree=.7,
        reg_alpha=.5, reg_lambda=2.0, objective="binary", random_state=seed,
        n_jobs=-1, verbose=-1,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight,
              eval_set=[(x_val, y_val)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])

    raw_val = model.predict_proba(x_val)[:, 1]
    eps = 1e-6
    val_clipped = np.clip(raw_val, eps, 1 - eps)
    calibrator = LogisticRegression(C=1.0).fit(
        np.log(val_clipped / (1 - val_clipped)).reshape(-1, 1), y_val)
    raw_all = model.predict_proba(x_all)[:, 1]
    all_clipped = np.clip(raw_all, eps, 1 - eps)
    probability = calibrator.predict_proba(np.log(all_clipped / (1 - all_clipped)).reshape(-1, 1))[:, 1]

    val_calibrated = probability[val.to_numpy()]
    threshold = threshold_for_f1(y_val.to_numpy(), val_calibrated)
    val_pr_auc = average_precision_score(y_val, val_calibrated)

    meta = df[["player_id", "game_date", "split", "role", "evaluation_cohort", "target_100d"]]
    model_name = f"gbm_seed{seed}"
    metrics = save_predictions_and_metrics(model_name, meta, probability, threshold)
    with open(RESULTS / f"{model_name}_model.pkl", "wb") as fh:
        pickle.dump({"model": model, "calibrator": calibrator, "feature_cols": feature_cols}, fh)
    pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(RESULTS / f"{model_name}_feature_importance.csv", index=False)
    write_run(model_name, {"seed": seed, "class_factor": class_factor, "threshold": threshold,
              "val_pr_auc": float(val_pr_auc), "best_iteration": int(model.best_iteration_),
              "n_input_features": len(feature_cols)})
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    main(a.seed)
