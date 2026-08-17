"""CPU main baseline: event-aware, cost-sensitive XGBoost."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from modeling_common import DATA, RESULTS, choose_threshold, load_flat, save_predictions_and_metrics, write_run


def main():
    RESULTS.mkdir(exist_ok=True)
    x, meta, columns = load_flat()
    labeled = meta.target_100d.notna()
    train = meta.split.eq("train") & labeled
    val = meta.split.eq("validation") & labeled
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train = imputer.fit_transform(x[train])
    x_val = imputer.transform(x[val])
    x_all = imputer.transform(x)
    y_train = meta.loc[train, "target_100d"].astype(int).to_numpy()
    y_val = meta.loc[val, "target_100d"].astype(int).to_numpy()
    event_weight = meta.loc[train, "event_weight"].fillna(1).to_numpy(float)
    effective_positive = event_weight[y_train == 1].sum()
    class_factor = float((y_train == 0).sum() / effective_positive)
    sample_weight = np.where(y_train == 1, event_weight * class_factor, 1.0)
    model = xgb.XGBClassifier(n_estimators=900, max_depth=5, learning_rate=.035,
        min_child_weight=8, subsample=.8, colsample_bytree=.7, reg_alpha=.2, reg_lambda=4,
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist", n_jobs=-1,
        random_state=42, early_stopping_rounds=70)
    model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_val, y_val)], verbose=False)
    raw_val = model.predict_proba(x_val)[:, 1]
    # Platt calibration uses Validation only; the selected map is frozen for Test.
    eps = 1e-6
    val_clipped = np.clip(raw_val, eps, 1-eps)
    calibrator = LogisticRegression(C=1.0).fit(np.log(val_clipped / (1-val_clipped)).reshape(-1, 1), y_val)
    raw_all = model.predict_proba(x_all)[:, 1]
    all_clipped = np.clip(raw_all, eps, 1-eps)
    probability = calibrator.predict_proba(np.log(all_clipped / (1-all_clipped)).reshape(-1, 1))[:, 1]
    threshold = choose_threshold(y_val, probability[val])
    metrics = save_predictions_and_metrics("xgboost_weighted", meta, probability, threshold)
    model.save_model(RESULTS / "xgboost_weighted_model.json")
    pd.DataFrame({"feature": columns, "importance": model.feature_importances_[:len(columns)]}).sort_values(
        "importance", ascending=False).to_csv(RESULTS / "xgboost_feature_importance.csv", index=False)
    write_run("xgboost_weighted", {"class_factor": class_factor, "threshold": threshold,
        "best_iteration": int(model.best_iteration), "n_input_features": len(columns)})
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
