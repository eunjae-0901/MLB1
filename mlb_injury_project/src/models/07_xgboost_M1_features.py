"""
항목 4(M0->M1) 비교용. 01_xgboost_bayesian.py와 완전히 동일한 절차(같은 상관관계
기반 feature selection, 같은 Bayesian Optimization 탐색 범위)인데, 03_build_
rolling_dataset.py에 새로 추가한 M1 변수(불펜: 3일/7일 등판수·투구수, 최근 14일
등판 간격 평균·최솟값, 연투·3연투 여부 / 선발: 직전경기·최근2경기·최근3경기최대
투구수, 짧은휴식 여부)까지 포함된 최신 data/processed/*.parquet을 사용한다는
점만 다르다.

numeric_feature_cols()가 parquet에 있는 모든 숫자 컬럼을 자동으로 잡기 때문에
코드 수정 없이 데이터만 다시 만들면 M1 변수가 자동으로 후보에 들어간다 - 그래서
01번을 그대로 복붙하되 결과 파일명만 "_M1"을 붙여 M0(01번) 결과와 구분한다
(model/, .gitignore 대상이라 덮어쓰면 M0 기준값을 잃어버리므로 파일명을 분리).

이진분류(binary)만, 두 역할(불펜/선발) 비교. 실행: python 07_xgboost_M1_features.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import CATEGORICAL_COLS, MODEL_DIR, load_role, numeric_feature_cols  # noqa: E402
from feature_selection import select_uncorrelated_features  # noqa: E402

CORR_THRESHOLD = 0.9
LABEL_MODE = "binary"
STAGE_TAG = "M1"

PBOUNDS = {
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 7),
    "reg_lambda": (0.5, 5.0),
}


def xgb_base_params(max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda):
    return dict(
        max_depth=int(round(max_depth)), learning_rate=learning_rate, n_estimators=2000,
        subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        reg_lambda=reg_lambda, tree_method="hist", enable_categorical=True,
        early_stopping_rounds=50, random_state=42,
        objective="binary:logistic", eval_metric="logloss",
    )


def auc_score(y, proba) -> float:
    return roc_auc_score(y, proba[:, 1])


def evaluate(model, X, y, name: str) -> float:
    proba = model.predict_proba(X)
    pred = model.predict(X)
    print(f"\n--- {name} (n={len(y):,}) ---")
    print(classification_report(y, pred, digits=3, zero_division=0))
    print("confusion matrix (행=실제, 열=예측):")
    print(confusion_matrix(y, pred))
    auc = auc_score(y, proba)
    print(f"AUC: {auc:.3f}")
    return auc


def run(role: str, n_iter: int = 25, init_points: int = 8):
    print(f"\n{'=' * 70}\n[{STAGE_TAG}] XGBoost + Bayesian Optimization - {role.upper()}\n{'=' * 70}")
    splits = load_role(role, exclude_other=True, binarize=True)
    all_num_cols = numeric_feature_cols(splits["train"])
    kept_num_cols = select_uncorrelated_features(splits["train"], all_num_cols, threshold=CORR_THRESHOLD)
    feature_cols = kept_num_cols + CATEGORICAL_COLS
    print(f"[입력변수] 후보 {len(all_num_cols)}개 -> 상관관계 제거 후 {len(kept_num_cols)}개 "
          f"(+ 범주형 {len(CATEGORICAL_COLS)}개)")

    xy = {s: (df[feature_cols], df["label"].astype(int)) for s, df in splits.items()}
    print(f"train={len(xy['train'][0]):,} val={len(xy['val'][0]):,} test={len(xy['test'][0]):,}")

    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    sample_weight = compute_sample_weight("balanced", y_train)

    def objective(max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, reg_lambda):
        params = xgb_base_params(max_depth, learning_rate, subsample, colsample_bytree,
                                  min_child_weight, reg_lambda)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)
        return auc_score(y_val, model.predict_proba(X_val))

    optimizer = BayesianOptimization(f=objective, pbounds=PBOUNDS, random_state=42, verbose=2)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = optimizer.max
    print(f"\n최적 하이퍼파라미터: {best['params']}")
    print(f"탐색 중 최고 val AUC: {best['target']:.4f}")

    trials_path = MODEL_DIR / f"07_{role}_xgboost_{LABEL_MODE}_{STAGE_TAG}_trials.json"
    trials_path.write_text(
        json.dumps([{"target": r["target"], "params": r["params"]} for r in optimizer.res],
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    p = best["params"]
    final_params = xgb_base_params(p["max_depth"], p["learning_rate"], p["subsample"],
                                    p["colsample_bytree"], p["min_child_weight"], p["reg_lambda"])
    X_test, y_test = xy["test"]

    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train, sample_weight=sample_weight, eval_set=[(X_val, y_val)], verbose=False)

    val_auc = evaluate(model, X_val, y_val, "Validation (최적 하이퍼파라미터)")
    test_auc = evaluate(model, X_test, y_test, "Test (최적 하이퍼파라미터)")

    model_path = MODEL_DIR / f"07_{role}_xgboost_{LABEL_MODE}_{STAGE_TAG}.json"
    model.save_model(model_path)

    summary_path = MODEL_DIR / f"07_{role}_xgboost_{LABEL_MODE}_{STAGE_TAG}_summary.json"
    summary_path.write_text(
        json.dumps({
            "role": role, "label_mode": LABEL_MODE, "stage": STAGE_TAG,
            "feature_cols": feature_cols, "n_candidate_cols": len(all_num_cols),
            "best_params": p, "val_auc": val_auc, "test_auc": test_auc,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {model_path}, {summary_path}")
    return val_auc, test_auc


def main():
    results = {}
    for role in ("bullpen", "starter"):
        val_auc, test_auc = run(role)
        results[role] = (val_auc, test_auc)

    print(f"\n{'=' * 70}\n[{STAGE_TAG}] 결과 요약 (M0 baseline: bullpen val/test 0.561/0.546, "
          f"starter val/test 0.563/0.522)\n{'=' * 70}")
    for role, (val_auc, test_auc) in results.items():
        print(f"{role}: val_auc={val_auc:.4f}  test_auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
