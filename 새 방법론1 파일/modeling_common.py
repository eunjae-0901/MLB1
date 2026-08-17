"""Shared preprocessing, evaluation, and output helpers."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    f1_score, fbeta_score, log_loss, matthews_corrcoef, precision_score, recall_score,
    roc_auc_score)

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"


def load_flat():
    z = np.load(DATA / "sequences_100d_5d.npz")
    meta = pd.read_parquet(DATA / "sequence_metadata.parquet")
    x = z["X"].astype("float32")
    names = z["feature_names"].astype(str).tolist()
    # Preserve temporal location for tree models by flattening each bin separately.
    flat = x.reshape(len(x), -1)
    columns = [f"lag{b*5:02d}_{name}" for b in range(x.shape[1]) for name in names]
    history = meta[["past_arm_il_count", "days_since_last_arm_il", "age"]].to_numpy("float32")
    flat = np.column_stack([flat, history])
    columns += ["past_arm_il_count", "days_since_last_arm_il", "age"]
    return flat, meta, columns


def choose_threshold(y, p):
    candidates = np.unique(np.quantile(p, np.linspace(.01, .99, 199)))
    scores = [fbeta_score(y, p >= t, beta=2, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def metric_row(y, p, threshold, **labels):
    pred = p >= threshold
    return {**labels, "n": len(y), "n_positive": int(np.sum(y)), "positive_rate": float(np.mean(y)),
        "threshold": float(threshold), "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred))}


def evaluation_masks(meta):
    return {
        "validation_all": meta.split.eq("validation"),
        "test_all": meta.split.eq("test"),
        "test_seen": meta.split.eq("test") & meta.evaluation_cohort.eq("seen_player"),
        "test_new": meta.split.eq("test") & meta.evaluation_cohort.eq("new_player"),
        "test_bullpen": meta.split.eq("test") & meta.role.eq("bullpen"),
        "test_starter": meta.split.eq("test") & meta.role.eq("starter"),
    }


def save_predictions_and_metrics(model_name, meta, probability, threshold):
    pred = meta[["player_id", "game_date", "split", "role", "evaluation_cohort", "target_100d"]].copy()
    pred["probability"] = probability
    pred["prediction"] = probability >= threshold
    pred.to_csv(RESULTS / f"{model_name}_predictions.csv", index=False)
    rows = []
    for cohort, mask in evaluation_masks(meta).items():
        mask = mask & meta.target_100d.notna()
        y = meta.loc[mask, "target_100d"].astype(int).to_numpy()
        rows.append(metric_row(y, probability[mask], threshold, model=model_name, cohort=cohort))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / f"{model_name}_metrics.csv", index=False)
    return metrics


def write_run(name, payload):
    (RESULTS / f"{name}_run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
