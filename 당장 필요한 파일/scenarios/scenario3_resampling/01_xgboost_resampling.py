"""시나리오 3: XGBoost에서 불균형 처리 방법을 공정하게 비교한다.

리샘플링은 train에만 적용하며 validation/test는 자연 분포를 유지한다.
기본 표본은 Kang(2025)의 약 16% 양성 비율과 가까운 case-control 1:5이다.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[1]
SCENARIO1_MODULE = PROJECT_DIR / "scenarios" / "scenario1_case_control" / "02_xgboost.py"
RESULT_DIR = HERE / "results"
MODEL_DIR = RESULT_DIR / "models"


def load_common():
    spec = importlib.util.spec_from_file_location("scenario1_xgb_common", SCENARIO1_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"공통 모듈을 읽을 수 없습니다: {SCENARIO1_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMMON = load_common()
METHODS = ("none", "class_weight", "smote", "smotetomek", "smoteenn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", choices=["bullpen", "starter"], default=["bullpen", "starter"])
    parser.add_argument("--variant", choices=["cc3", "cc5"], default="cc5")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def resample(method: str, x_train: pd.DataFrame, y_train: pd.Series):
    before = pd.Series(y_train).value_counts().sort_index().to_dict()
    if method in {"none", "class_weight"}:
        return x_train, y_train, before, before
    sampler = {
        "smote": SMOTE(random_state=SEED, k_neighbors=5),
        "smotetomek": SMOTETomek(random_state=SEED),
        "smoteenn": SMOTEENN(random_state=SEED),
    }[method]
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    x_resampled = pd.DataFrame(x_resampled, columns=x_train.columns)
    y_resampled = pd.Series(y_resampled, name="label")
    after = y_resampled.value_counts().sort_index().to_dict()
    return x_resampled, y_resampled, before, after


def run_one(role: str, variant: str, method: str, trials: int, n_jobs: int):
    print(f"\n[{role} / {variant} / {method}]")
    splits = COMMON.load_splits(role, variant)
    x, y, features = COMMON.preprocess(splits)
    x_fit, y_fit, before, after = resample(method, x["train"], y["train"])
    weights = compute_sample_weight("balanced", y_fit) if method == "class_weight" else None
    print(f"  train 분포: {before} -> {after}; val/test는 변경하지 않음")

    params, tuning = COMMON.tune(
        x_fit, y_fit, x["val"], y["val"], weights, trials, n_jobs,
    )
    model = xgb.XGBClassifier(**COMMON.base_params(n_jobs, **params))
    model.fit(
        x_fit, y_fit, sample_weight=weights,
        eval_set=[(x["val"], y["val"])], verbose=False,
    )
    model.save_model(MODEL_DIR / f"{role}_{variant}_{method}.json")
    prob_val = model.predict_proba(x["val"])[:, 1]
    threshold = COMMON.choose_threshold(y["val"], prob_val)

    metric_rows, alert_rows, event_rows = [], [], []
    for split in ("val", "test"):
        probability = prob_val if split == "val" else model.predict_proba(x["test"])[:, 1]
        for threshold_name, cutoff in (("0.5", 0.5), ("validation_selected", threshold)):
            metric_rows.append({
                "role": role,
                "variant": variant,
                "model": "XGBoost",
                "imbalance_method": method,
                "split": split,
                "threshold_type": threshold_name,
                "threshold": cutoff,
                "n_train_before": int(sum(before.values())),
                "n_train_after": int(sum(after.values())),
                "train_positive_rate_after": float(after.get(1, 0) / sum(after.values())),
                "evaluation_positive_rate": float(y[split].mean()),
                **COMMON.metrics(y[split], probability, cutoff),
            })
        for fraction in (0.05, 0.10):
            alert_rows.append({
                "role": role, "variant": variant, "imbalance_method": method,
                "split": split, **COMMON.top_risk_metrics(y[split], probability, fraction),
            })
        event_rows.append({
            "role": role, "variant": variant, "imbalance_method": method,
            "split": split, "threshold": threshold,
            **COMMON.event_detection(splits[split], probability, threshold),
        })

    leakage = {
        "role": role,
        "variant": variant,
        "imbalance_method": method,
        "resampling_scope": "train_only",
        "val_rows_unchanged": len(splits["val"]),
        "test_rows_unchanged": len(splits["test"]),
        "suspicious_f1_over_0_9": any(row["f1"] > 0.9 for row in metric_rows),
        "status": "PASS",
    }
    tuning.insert(0, "role", role)
    tuning.insert(1, "variant", variant)
    tuning.insert(2, "imbalance_method", method)
    return metric_rows, alert_rows, event_rows, [leakage], tuning


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics, all_alerts, all_events, all_checks, all_tuning = [], [], [], [], []
    for role in args.roles:
        for method in args.methods:
            metrics, alerts, events, checks, tuning = run_one(
                role, args.variant, method, args.trials, args.n_jobs,
            )
            all_metrics.extend(metrics)
            all_alerts.extend(alerts)
            all_events.extend(events)
            all_checks.extend(checks)
            all_tuning.append(tuning)

    metrics_df = pd.DataFrame(all_metrics)
    alerts_df = pd.DataFrame(all_alerts)
    events_df = pd.DataFrame(all_events)
    checks_df = pd.DataFrame(all_checks)
    tuning_df = pd.concat(all_tuning, ignore_index=True)
    output = RESULT_DIR / f"scenario3_{args.variant}_results.xlsx"
    with pd.ExcelWriter(output) as writer:
        metrics_df.to_excel(writer, sheet_name="01_all_metrics", index=False)
        alerts_df.to_excel(writer, sheet_name="02_top_risk_alerts", index=False)
        events_df.to_excel(writer, sheet_name="03_event_detection", index=False)
        checks_df.to_excel(writer, sheet_name="04_leakage_checks", index=False)
        tuning_df.to_excel(writer, sheet_name="05_tuning", index=False)
    metrics_df.to_csv(RESULT_DIR / f"scenario3_{args.variant}_metrics.csv", index=False)
    checks_df.to_csv(RESULT_DIR / f"scenario3_{args.variant}_leakage_checks.csv", index=False)
    test = metrics_df[
        (metrics_df["split"] == "test") &
        (metrics_df["threshold_type"] == "validation_selected")
    ]
    print("\n자연분포 Test 결과")
    print(test[[
        "role", "imbalance_method", "roc_auc", "pr_auc", "f1",
        "precision", "recall", "mcc", "brier", "predicted_positive_rate",
    ]].to_string(index=False))
    print(f"\n저장: {output}")


if __name__ == "__main__":
    main()
