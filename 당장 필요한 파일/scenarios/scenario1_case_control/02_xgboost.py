"""시나리오 1 XGBoost 학습 및 자연분포 validation/test 평가.

핵심 원칙:
- case-control은 train에만 적용한다.
- hyperparameter와 threshold는 validation에서만 선택한다.
- test는 선택이 끝난 모델을 마지막에 평가한다.
- threshold 0.5와 validation-selected threshold 결과를 모두 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
SCENARIO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCENARIO_DIR.parents[1]
DATA_DIR = PROJECT_DIR / "data"
CC_DIR = DATA_DIR / "case_control"
RESULT_DIR = SCENARIO_DIR / "results"
MODEL_DIR = RESULT_DIR / "models"
FIGURE_DIR = RESULT_DIR / "figures"

ID_COLS = {
    "player_id", "window_end_date", "il_start_date", "injury_class_strict",
    "days_to_injury", "split", "label",
}
CATEGORICAL_COLS = ("p_throws", "birth_country")
DEFAULT_ROLES = ("bullpen", "starter")
DEFAULT_VARIANTS = ("baseline", "cc3", "cc5", "cc3_capped3")

PARAM_SPACE = {
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.02, 0.04, 0.06, 0.1],
    "subsample": [0.65, 0.8, 1.0],
    "colsample_bytree": [0.65, 0.8, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", nargs="+", choices=DEFAULT_ROLES, default=list(DEFAULT_ROLES))
    parser.add_argument("--variants", nargs="+", choices=DEFAULT_VARIANTS, default=list(DEFAULT_VARIANTS))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def dataset_path(role: str, variant: str) -> Path:
    if variant == "baseline":
        return DATA_DIR / f"{role}_dataset.csv"
    return CC_DIR / f"{role}_dataset_{variant}.csv"


def load_splits(role: str, variant: str) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(dataset_path(role, variant))
    df = df.loc[df["label"] != 3].copy()
    df["label"] = (df["label"] > 0).astype("int8")
    for col in ("window_end_date", "il_start_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return {name: df.loc[df["split"] == name].reset_index(drop=True) for name in ("train", "val", "test")}


def select_numeric_features(train: pd.DataFrame, threshold: float = 0.90) -> list[str]:
    candidates = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c not in ID_COLS
    ]
    corr = train[candidates].corr().abs()
    kept: list[str] = []
    for col in candidates:
        if not any(pd.notna(corr.loc[col, prev]) and corr.loc[col, prev] > threshold for prev in kept):
            kept.append(col)
    return kept


def preprocess(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], list[str]]:
    numeric = select_numeric_features(splits["train"])
    median = splits["train"][numeric].median()
    category_levels = {
        col: sorted(splits["train"][col].fillna("__MISSING__").astype(str).unique())
        for col in CATEGORICAL_COLS
    }
    encoded: dict[str, pd.DataFrame] = {}
    labels: dict[str, pd.Series] = {}
    train_columns: list[str] | None = None
    for split, df in splits.items():
        x_num = df[numeric].replace([np.inf, -np.inf], np.nan).fillna(median).astype("float32")
        cat_parts = []
        for col in CATEGORICAL_COLS:
            raw = df[col].fillna("__MISSING__").astype(str)
            raw = raw.where(raw.isin(category_levels[col]), "__UNKNOWN__")
            levels = [*category_levels[col], "__UNKNOWN__"]
            values = pd.Categorical(
                raw,
                categories=levels,
            )
            cat_parts.append(pd.get_dummies(values, prefix=col, dtype="float32"))
        x = pd.concat([x_num.reset_index(drop=True), *cat_parts], axis=1)
        if train_columns is None:
            train_columns = list(x.columns)
        else:
            x = x.reindex(columns=train_columns, fill_value=0.0)
        encoded[split] = x
        labels[split] = df["label"].astype(int)
    assert train_columns is not None
    return encoded, labels, train_columns


def base_params(n_jobs: int, **overrides: float | int) -> dict:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "n_estimators": 1200,
        "early_stopping_rounds": 50,
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": n_jobs,
    }
    params.update(overrides)
    return params


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(y_true: Iterable[int], probability: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true)
    pred = (probability >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, probability),
        "pr_auc": average_precision_score(y, probability),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, probability),
        "predicted_positive_rate": float(pred.mean()),
    }


def top_risk_metrics(y_true: Iterable[int], probability: np.ndarray, fraction: float) -> dict[str, float]:
    y = np.asarray(y_true)
    k = max(1, int(np.ceil(len(y) * fraction)))
    idx = np.argsort(-probability)[:k]
    detected = int(y[idx].sum())
    positives = int(y.sum())
    return {
        "top_fraction": fraction,
        "alerts": k,
        "detected_rows": detected,
        "recall_at_top": detected / positives if positives else np.nan,
        "precision_at_top": detected / k,
    }


def event_detection(
    meta: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    positive = meta.loc[meta["label"] == 1].copy()
    if positive.empty:
        return {"unique_events": 0, "detected_events": 0, "event_detection_rate": np.nan, "median_lead_days": np.nan}
    positive["probability"] = probability[positive.index]
    positive["alert"] = positive["probability"] >= threshold
    positive["event_id"] = (
        positive["player_id"].astype(str) + "_" +
        positive["il_start_date"].astype(str) + "_" +
        positive["injury_class_strict"].astype(str)
    )
    grouped = positive.groupby("event_id", dropna=False)
    detected = grouped["alert"].any()
    lead = positive.loc[positive["alert"]].groupby("event_id")["days_to_injury"].max()
    return {
        "unique_events": int(len(detected)),
        "detected_events": int(detected.sum()),
        "event_detection_rate": float(detected.mean()),
        "median_lead_days": float(lead.median()) if len(lead) else np.nan,
    }


def tune(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray | None,
    trials: int,
    n_jobs: int,
) -> tuple[dict, pd.DataFrame]:
    rows = []
    best_score = -np.inf
    best_params: dict = {}
    for trial, params in enumerate(ParameterSampler(PARAM_SPACE, n_iter=trials, random_state=SEED), start=1):
        model = xgb.XGBClassifier(**base_params(n_jobs, **params))
        model.fit(
            x_train, y_train, sample_weight=sample_weight,
            eval_set=[(x_val, y_val)], verbose=False,
        )
        prob = model.predict_proba(x_val)[:, 1]
        score = average_precision_score(y_val, prob)
        rows.append({"trial": trial, "val_pr_auc": score, "best_iteration": model.best_iteration, **params})
        print(f"    trial {trial:02d}/{trials}: val PR-AUC={score:.5f}")
        if score > best_score:
            best_score, best_params = score, params
    return best_params, pd.DataFrame(rows)


def save_plots(predictions: pd.DataFrame, role: str, variant: str, weight_name: str) -> None:
    test = predictions.loc[predictions["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    precision, recall, _ = precision_recall_curve(test["y_true"], test["probability"])
    axes[0].plot(recall, precision)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve")
    axes[0].grid(alpha=0.25)
    frac_pos, mean_pred = calibration_curve(test["y_true"], test["probability"], n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, marker="o", label="model")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    axes[1].set(xlabel="Mean predicted risk", ylabel="Observed rate", title="Test calibration")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"{role} / {variant} / {weight_name}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{role}_{variant}_{weight_name}.png", dpi=180)
    plt.close(fig)


def run_one(
    role: str,
    variant: str,
    use_class_weight: bool,
    trials: int,
    n_jobs: int,
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame, pd.DataFrame]:
    print(f"\n[{role} / {variant} / class_weight={use_class_weight}]")
    splits = load_splits(role, variant)
    x, y, feature_names = preprocess(splits)
    weights = compute_sample_weight("balanced", y["train"]) if use_class_weight else None
    best_params, tuning = tune(x["train"], y["train"], x["val"], y["val"], weights, trials, n_jobs)

    model = xgb.XGBClassifier(**base_params(n_jobs, **best_params))
    model.fit(x["train"], y["train"], sample_weight=weights, eval_set=[(x["val"], y["val"])], verbose=False)
    weight_name = "weighted" if use_class_weight else "unweighted"
    model.save_model(MODEL_DIR / f"{role}_{variant}_{weight_name}.json")
    (MODEL_DIR / f"{role}_{variant}_{weight_name}_features.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    probabilities = {split: model.predict_proba(x[split])[:, 1] for split in ("val", "test")}
    selected_threshold = choose_threshold(y["val"], probabilities["val"])
    metric_rows, alert_rows, event_rows, prediction_parts = [], [], [], []
    for split in ("val", "test"):
        for threshold_name, threshold in (("0.5", 0.5), ("validation_selected", selected_threshold)):
            row = metrics(y[split], probabilities[split], threshold)
            metric_rows.append({
                "role": role, "variant": variant, "model": "XGBoost",
                "imbalance_method": weight_name, "split": split,
                "threshold_type": threshold_name, "threshold": threshold,
                "n_rows": len(y[split]), "positive_rate": float(y[split].mean()), **row,
            })
            if threshold_name == "validation_selected":
                event_rows.append({
                    "role": role, "variant": variant, "split": split,
                    "threshold": threshold, **event_detection(splits[split], probabilities[split], threshold),
                })
        for fraction in (0.05, 0.10):
            alert_rows.append({
                "role": role, "variant": variant, "split": split,
                **top_risk_metrics(y[split], probabilities[split], fraction),
            })
        meta = splits[split][["player_id", "window_end_date", "il_start_date", "injury_class_strict", "days_to_injury"]].copy()
        meta.insert(0, "split", split)
        meta["y_true"] = y[split].to_numpy()
        meta["probability"] = probabilities[split]
        meta["prediction_0_5"] = (probabilities[split] >= 0.5).astype(int)
        meta["prediction_val_threshold"] = (probabilities[split] >= selected_threshold).astype(int)
        prediction_parts.append(meta)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    save_plots(predictions, role, variant, weight_name)
    tuning.insert(0, "role", role)
    tuning.insert(1, "variant", variant)
    tuning.insert(2, "imbalance_method", weight_name)
    print(f"  selected threshold={selected_threshold:.5f}, best params={best_params}")
    return metric_rows, alert_rows, event_rows, predictions, tuning


def main() -> None:
    args = parse_args()
    for directory in (RESULT_DIR, MODEL_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    use_weight = not args.no_class_weight
    metric_rows, alert_rows, event_rows = [], [], []
    predictions, tuning_rows = [], []

    for role in args.roles:
        for variant in args.variants:
            result = run_one(role, variant, use_weight, args.trials, args.n_jobs)
            metrics_part, alerts_part, events_part, predictions_part, tuning_part = result
            metric_rows.extend(metrics_part)
            alert_rows.extend(alerts_part)
            event_rows.extend(events_part)
            predictions_part.insert(0, "role", role)
            predictions_part.insert(1, "variant", variant)
            predictions.append(predictions_part)
            tuning_rows.append(tuning_part)

    metrics_df = pd.DataFrame(metric_rows)
    alerts_df = pd.DataFrame(alert_rows)
    events_df = pd.DataFrame(event_rows)
    predictions_df = pd.concat(predictions, ignore_index=True)
    tuning_df = pd.concat(tuning_rows, ignore_index=True)
    suffix = "weighted" if use_weight else "unweighted"
    metrics_df.to_csv(RESULT_DIR / f"scenario1_metrics_{suffix}.csv", index=False)
    alerts_df.to_csv(RESULT_DIR / f"scenario1_top_risk_{suffix}.csv", index=False)
    events_df.to_csv(RESULT_DIR / f"scenario1_event_detection_{suffix}.csv", index=False)
    predictions_df.to_csv(RESULT_DIR / f"scenario1_predictions_{suffix}.csv", index=False)
    tuning_df.to_csv(RESULT_DIR / f"scenario1_tuning_{suffix}.csv", index=False)
    with pd.ExcelWriter(RESULT_DIR / f"scenario1_results_{suffix}.xlsx") as writer:
        metrics_df.to_excel(writer, sheet_name="01_all_metrics", index=False)
        alerts_df.to_excel(writer, sheet_name="02_top_risk_alerts", index=False)
        events_df.to_excel(writer, sheet_name="03_event_detection", index=False)
        tuning_df.to_excel(writer, sheet_name="04_tuning", index=False)
    print("\nTest 결과")
    print(metrics_df.loc[metrics_df["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
