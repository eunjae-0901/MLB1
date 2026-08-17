"""Methodology 05b: validation-fit logistic stacking over lstm/vit/resnet/
xgboost/gbm (05 engineered-feature model). Blend weights are fit on
Validation only and frozen before scoring Test - Test is never touched
by fitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from modeling_common import DATA, RESULTS, save_predictions_and_metrics

meta = pd.read_parquet(DATA / "sequence_metadata.parquet")
n = len(meta)


def gpu_model_probs(name: str, seed: int) -> np.ndarray:
    p = pd.read_csv(RESULTS / f"{name}_seed{seed}_predictions.csv")
    out = np.full(n, np.nan)
    val = p[p.cohort.eq("validation_all")]
    test = p[p.cohort.eq("test_all")]
    out[val.row_index.to_numpy()] = val.probability.to_numpy()
    out[test.row_index.to_numpy()] = test.probability.to_numpy()
    return out


def flat_model_probs(name: str) -> np.ndarray:
    p = pd.read_csv(RESULTS / f"{name}_predictions.csv")
    assert len(p) == n
    return p["probability"].to_numpy()


def main():
    columns = {
        "lstm": np.mean([gpu_model_probs("lstm", s) for s in (42, 52, 62)], axis=0),
        "vit": np.mean([gpu_model_probs("vit", s) for s in (42, 52, 62)], axis=0),
        "resnet": np.mean([gpu_model_probs("resnet", s) for s in (42, 52, 62)], axis=0),
        "xgboost": flat_model_probs("xgboost_weighted"),
        "gbm": np.mean([flat_model_probs(f"gbm_seed{s}") for s in (42, 52, 62)], axis=0),
    }
    x = pd.DataFrame(columns)
    labeled = meta.target_100d.notna()
    val = labeled & meta.split.eq("validation")

    x_val = x.loc[val].to_numpy()
    y_val = meta.loc[val, "target_100d"].astype(int).to_numpy()
    blender = LogisticRegression(C=1.0).fit(x_val, y_val)
    print("blend weights:", dict(zip(x.columns, blender.coef_[0].round(3))))

    x_valid_rows = x.notna().all(axis=1)
    probability = np.full(n, np.nan)
    probability[x_valid_rows] = blender.predict_proba(x.loc[x_valid_rows].to_numpy())[:, 1]

    from modeling_common import choose_threshold
    threshold = choose_threshold(y_val, probability[val.to_numpy()])
    metrics = save_predictions_and_metrics("stack_ensemble", meta, probability, threshold)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
