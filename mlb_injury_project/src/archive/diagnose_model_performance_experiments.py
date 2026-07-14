"""
1단계 진단: XGBoost 성능이 낮게 나온 원인을 좁히기 위한 실험 비교.

중요: 여기서는 val set만 본다. test는 최종 모델을 하나 확정한 뒤 딱 한 번만 확인한다
(여러 번 test를 보면서 고르면 test에 은근히 과적합되기 때문).

비교 축:
  - 다중분류(0/1/2/3) vs 이진분류(0 vs 나머지)
  - birth_country 포함 vs 제외
  - balanced class weight vs 완화된(sqrt) weight
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DROP_COLS = [
    "player_id", "window_end_date", "il_start_date",
    "injury_class_strict", "days_to_injury", "split", "label",
]
CATEGORICAL_COLS = ["p_throws", "birth_country"]

BASE_PARAMS = dict(
    max_depth=5,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    early_stopping_rounds=50,
    random_state=42,
)


def load_role(role: str):
    df = pd.read_parquet(PROCESSED_DIR / f"{role}_window_dataset.parquet")
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    return df


def run_experiment(name, train_df, val_df, binary, drop_birth_country, weight_mode):
    drop_cols = list(DROP_COLS)
    if drop_birth_country:
        drop_cols = drop_cols  # birth_country는 아래서 별도로 X에서 제거

    X_train = train_df.drop(columns=DROP_COLS).copy()
    X_val = val_df.drop(columns=DROP_COLS).copy()
    if drop_birth_country:
        X_train = X_train.drop(columns=["birth_country"])
        X_val = X_val.drop(columns=["birth_country"])

    if binary:
        y_train = (train_df["label"] > 0).astype(int)
        y_val = (val_df["label"] > 0).astype(int)
        objective, num_class = "binary:logistic", None
        eval_metric = "logloss"
    else:
        y_train = train_df["label"].astype(int)
        y_val = val_df["label"].astype(int)
        objective, num_class = "multi:softprob", 4
        eval_metric = "mlogloss"

    balanced = compute_sample_weight("balanced", y_train)
    if weight_mode == "balanced":
        sample_weight = balanced
    elif weight_mode == "sqrt":
        # balanced weight를 그대로 쓰면 희귀 클래스가 지나치게 과대반영될 수 있어
        # 제곱근으로 완화 (예: 33배 -> 5.7배)
        sample_weight = np.sqrt(balanced)
    else:
        sample_weight = None

    params = dict(BASE_PARAMS)
    params["objective"] = objective
    params["eval_metric"] = eval_metric
    if num_class:
        params["num_class"] = num_class

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    proba = model.predict_proba(X_val)
    pred = model.predict(X_val)

    if binary:
        auc = roc_auc_score(y_val, proba[:, 1])
    else:
        auc = roc_auc_score(y_val, proba, average="macro", multi_class="ovr")

    report = classification_report(y_val, pred, digits=3, zero_division=0, output_dict=True)
    if binary:
        minority_recall = report["1"]["recall"]
    else:
        minority_recall = np.mean([report[c]["recall"] for c in ("1", "2") if c in report])

    print(f"\n[{name}] best_iter={model.best_iteration}  AUC={auc:.3f}  "
          f"부상클래스 평균recall={minority_recall:.3f}")
    return {"name": name, "auc": auc, "minority_recall": minority_recall}


def run_shoulder_elbow_only(name, train_df, val_df, weight_mode="balanced"):
    """'그 외'(label=3) 행을 아예 제거하고 어깨/팔꿈치(1,2) vs 안다침(0)만 비교."""
    train_f = train_df[train_df["label"] != 3]
    val_f = val_df[val_df["label"] != 3]
    return run_experiment(
        name, train_f, val_f, binary=True, drop_birth_country=False,
        weight_mode=weight_mode,
    )


def main():
    for role in ("bullpen", "starter"):
        df = load_role(role)
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        print(f"\n{'#' * 70}\n{role.upper()}  train={len(train_df):,} val={len(val_df):,}")

        results = []
        if role == "bullpen":
            results.append(run_experiment(
                "A. baseline (multiclass, all feat, balanced weight)",
                train_df, val_df, binary=False, drop_birth_country=False, weight_mode="balanced"))
            results.append(run_experiment(
                "B. binary (0 vs injury), all feat, balanced weight",
                train_df, val_df, binary=True, drop_birth_country=False, weight_mode="balanced"))
            results.append(run_experiment(
                "C. multiclass, drop birth_country, balanced weight",
                train_df, val_df, binary=False, drop_birth_country=True, weight_mode="balanced"))
            results.append(run_experiment(
                "D. multiclass, all feat, sqrt-softened weight",
                train_df, val_df, binary=False, drop_birth_country=False, weight_mode="sqrt"))
            results.append(run_experiment(
                "E. binary, drop birth_country, balanced weight",
                train_df, val_df, binary=True, drop_birth_country=True, weight_mode="balanced"))

        excluded = (train_df["label"] == 3).sum() + (val_df["label"] == 3).sum()
        print(f"'그 외'(label=3) 제외 행 수: {excluded:,}")
        results.append(run_shoulder_elbow_only(
            "G. 어깨/팔꿈치 vs 안다침 ('그 외' 행 제거), balanced weight",
            train_df, val_df, weight_mode="balanced"))
        results.append(run_shoulder_elbow_only(
            "H. 어깨/팔꿈치 vs 안다침 ('그 외' 행 제거), sqrt weight",
            train_df, val_df, weight_mode="sqrt"))

        print("\n" + "=" * 70)
        print(f"{role} 요약")
        print("=" * 70)
        summary = pd.DataFrame(results)
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
