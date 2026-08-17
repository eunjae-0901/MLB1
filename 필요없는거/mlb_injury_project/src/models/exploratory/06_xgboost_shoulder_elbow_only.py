"""
모델 6. XGBoost, '그 외'(label=3) 행을 아예 제거하고 0(안다침)/1(어깨)/2(팔꿈치)
3종 분류. 모델 2(02_xgboost_no_other.py)와 다른 점: 모델 2는 어깨·팔꿈치를
1로 합쳐서 이진분류하지만, 이 모델은 셋을 그대로 둬서 어깨/팔꿈치를 구분해서 맞힌다.

'그 외'가 허리/무릎/손/질병 등 서로 무관한 부상을 뭉뚱그린 잡동사니 카테고리라
학습에 방해가 될 수 있다는 가설을 검증하기 위한 버전.
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
    print(f"\n{'=' * 60}\n[모델6] XGBoost 3종분류(그 외 제외, 어깨/팔꿈치 구분) - {role.upper()}\n{'=' * 60}")
    splits = load_role(role, exclude_other=True, binarize=False)
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

    model.save_model(MODEL_DIR / f"06_{role}_xgboost_shoulder_elbow_only.json")


def main():
    for role in ("bullpen", "starter"):
        train_role(role)


if __name__ == "__main__":
    main()
