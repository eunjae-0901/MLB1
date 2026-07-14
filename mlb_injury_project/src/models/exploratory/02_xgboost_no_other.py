"""
모델 2. XGBoost, '그 외'(label=3) 행을 제거하고 이진분류(0=안다침 / 1=어깨·팔꿈치)
입력: data/processed/{role}_window_dataset.parquet (rolling-window 평균 요약)

'그 외'는 COVID 격리·질병·부위불명 등 투구 패턴과 무관한 사유가 섞여있어 신호를
희석시킬 수 있다는 가설을 검증하기 위한 버전. (진단 실험에서 선발은 개선,
불펜은 큰 차이 없었음 - 두 역할 다 동일 조건으로 재확인)
"""
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_xgb import MODEL_DIR, evaluate, load_role, prepare_xy  # noqa: E402

XGB_PARAMS = dict(
    objective="binary:logistic",
    max_depth=5,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42,
)


def train_role(role: str):
    print(f"\n{'=' * 60}\n[모델2] XGBoost 이진분류(그 외 제외) - {role.upper()}\n{'=' * 60}")
    splits = load_role(role, exclude_other=True)
    X_train, y_train = prepare_xy(splits["train"])
    X_val, y_val = prepare_xy(splits["val"])
    X_test, y_test = prepare_xy(splits["test"])
    print(f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")
    print(f"train 라벨 분포: {y_train.value_counts().to_dict()}")

    sample_weight = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_val, y_val)], verbose=False)
    print(f"best_iteration={model.best_iteration}")

    evaluate(model, X_val, y_val, "Validation")
    evaluate(model, X_test, y_test, "Test")

    model.save_model(MODEL_DIR / f"02_{role}_xgboost_no_other.json")


def main():
    for role in ("bullpen", "starter"):
        train_role(role)


if __name__ == "__main__":
    main()
