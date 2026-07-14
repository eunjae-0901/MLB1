"""
모델 8. XGBoost, '그 외'(label=3) 행을 삭제하지 않고 그대로 두되, 라벨만 0으로
합쳐서 0(안다침+그 외 부상)/1(어깨)/2(팔꿈치) 3종 분류.

모델 6(06_xgboost_shoulder_elbow_only.py)과의 차이: 모델 6은 '그 외' 행을 아예
삭제해서 표본 수 자체가 줄어드는데(불펜 74,686건), 이 모델은 행을 하나도 안 지우고
전체 표본(불펜 135,743건)을 그대로 쓰면서 '그 외 부상'을 "어깨·팔꿈치는 안 다쳤다"는
정보로만 활용한다.
"""
import sys
from pathlib import Path

import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_xgb import MODEL_DIR, evaluate, load_role, prepare_xy  # noqa: E402

XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="mlogloss",
    early_stopping_rounds=50,
    random_state=42,
)


def train_role(role: str):
    print(f"\n{'=' * 60}\n[모델8] XGBoost 3종분류(그 외 -> 0으로 병합) - {role.upper()}\n{'=' * 60}")
    splits = load_role(role, merge_other_into_healthy=True)
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

    model.save_model(MODEL_DIR / f"08_{role}_xgboost_merge_other.json")


def main():
    for role in ("bullpen", "starter"):
        train_role(role)


if __name__ == "__main__":
    main()
