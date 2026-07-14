"""
모델 1 (공식 파이프라인 1단계). XGBoost + Bayesian Optimization, rolling-window
데이터(bullpen/starter_window_dataset.parquet). '그 외'(label=3) 행은 항상 제외하고,
--label_mode로 두 가지 분류 방식을 둘 다 지원한다.
  3class : 0(안다침)/1(어깨)/2(팔꿈치) 3종 분류
  binary : 1과 2를 합쳐서 0(안다침) vs 1(어깨 또는 팔꿈치) 이진분류

교수님 지시사항 3~5번 반영:
  3) 데이터 전처리 -> 학습 -> 평가까지 전체 파이프라인이 정상 작동하는지 확인
     (최고 성능이 목표가 아니라 파이프라인 검증 + 초기 성능 산출이 목표)
  4) 주요 하이퍼파라미터 6개를 골라 탐색 범위를 정하고 Bayesian Optimization 수행
  5) 최적 하이퍼파라미터와 성능지표를 함께 기록, 이후 다른 모델과 비교

탐색 대상 하이퍼파라미터 6개(4~7개 권장 범위 안에서 선정)와 범위는 XGBoost 공식 문서
및 여러 튜닝 가이드(xgboosting.com "Suggested Ranges for Tuning XGBoost
Hyperparameters", Kaggle Bayesian-Optimization-with-XGBoost 튜토리얼 등)에서 흔히
쓰이는 값을 참고해 정했다.
  max_depth        (3, 10)   : 트리 깊이. 너무 깊으면 과적합, 너무 얕으면 과소적합.
  learning_rate    (0.01, 0.3): 각 트리 반영 비율. 작을수록 안정적이지만 느림.
  subsample        (0.5, 1.0): 트리마다 샘플링할 행 비율. 낮추면 과적합 방지.
  colsample_bytree (0.5, 1.0): 트리마다 샘플링할 컬럼 비율. 낮추면 과적합 방지.
  min_child_weight (1, 7)    : 리프 노드가 가지는 최소 샘플 가중치 합. 클수록 보수적.
  reg_lambda       (0.5, 5.0): L2 정규화 강도.

입력변수 정리: 원래 rolling-window 숫자 컬럼이 56개인데, 구종 그룹별 지표들끼리
상관관계가 매우 높은 경우가 많아(예: 전체 평균 구속과 직구 평균 구속) 상관계수
0.9 초과인 컬럼은 feature_selection.select_uncorrelated_features로 미리 제거한다.

실행: python 01_xgboost_bayesian.py --role both --label_mode 3class
      python 01_xgboost_bayesian.py --role both --label_mode binary
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

PBOUNDS = {
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "min_child_weight": (1, 7),
    "reg_lambda": (0.5, 5.0),
}


def xgb_base_params(label_mode: str, max_depth, learning_rate, subsample, colsample_bytree,
                     min_child_weight, reg_lambda):
    common = dict(
        max_depth=int(round(max_depth)), learning_rate=learning_rate, n_estimators=2000,
        subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        reg_lambda=reg_lambda, tree_method="hist", enable_categorical=True,
        early_stopping_rounds=50, random_state=42,
    )
    if label_mode == "binary":
        return dict(objective="binary:logistic", eval_metric="logloss", **common)
    return dict(objective="multi:softprob", num_class=3, eval_metric="mlogloss", **common)


def auc_score(y, proba, label_mode: str) -> float:
    if label_mode == "binary":
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, average="macro", multi_class="ovr")


def build_xy(splits: dict, feature_cols: list[str]):
    xy = {}
    for s, df in splits.items():
        xy[s] = (df[feature_cols], df["label"].astype(int))
    return xy


def evaluate(model, X, y, name: str, label_mode: str) -> float:
    proba = model.predict_proba(X)
    pred = model.predict(X)
    print(f"\n--- {name} (n={len(y):,}) ---")
    print(classification_report(y, pred, digits=3, zero_division=0))
    print("confusion matrix (행=실제, 열=예측):")
    print(confusion_matrix(y, pred))
    try:
        auc = auc_score(y, proba, label_mode)
        print(f"AUC: {auc:.3f}")
    except ValueError as e:
        auc = float("nan")
        print(f"AUC 계산 불가: {e}")
    return auc


def make_objective(xy, label_mode: str):
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    sample_weight = compute_sample_weight("balanced", y_train)

    def objective(max_depth, learning_rate, subsample, colsample_bytree,
                  min_child_weight, reg_lambda):
        params = xgb_base_params(label_mode, max_depth, learning_rate, subsample,
                                  colsample_bytree, min_child_weight, reg_lambda)
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)
        try:
            return auc_score(y_val, proba, label_mode)
        except ValueError:
            return 0.0

    return objective


def run(role: str, label_mode: str, n_iter: int, init_points: int):
    print(f"\n{'=' * 70}\n[모델1:{label_mode}] XGBoost + Bayesian Optimization - {role.upper()}\n{'=' * 70}")
    splits = load_role(role, exclude_other=True, binarize=(label_mode == "binary"))
    all_num_cols = numeric_feature_cols(splits["train"])
    kept_num_cols = select_uncorrelated_features(splits["train"], all_num_cols, threshold=CORR_THRESHOLD)
    feature_cols = kept_num_cols + CATEGORICAL_COLS

    xy = build_xy(splits, feature_cols)
    print(f"train={len(xy['train'][0]):,} val={len(xy['val'][0]):,} test={len(xy['test'][0]):,}")
    print(f"train 라벨 분포: {xy['train'][1].value_counts().to_dict()}")

    objective = make_objective(xy, label_mode)
    optimizer = BayesianOptimization(f=objective, pbounds=PBOUNDS, random_state=42, verbose=2)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best = optimizer.max
    print(f"\n최적 하이퍼파라미터: {best['params']}")
    print(f"탐색 중 최고 val AUC: {best['target']:.4f}")

    trials_path = MODEL_DIR / f"01_{role}_xgboost_bayesian_{label_mode}_trials.json"
    trials_path.write_text(
        json.dumps([{"target": r["target"], "params": r["params"]} for r in optimizer.res],
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {trials_path}")

    p = best["params"]
    final_params = xgb_base_params(label_mode, p["max_depth"], p["learning_rate"], p["subsample"],
                                    p["colsample_bytree"], p["min_child_weight"], p["reg_lambda"])
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    sample_weight = compute_sample_weight("balanced", y_train)

    model = xgb.XGBClassifier(**final_params)
    model.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_val, y_val)], verbose=False)
    print(f"best_iteration={model.best_iteration}")

    val_auc = evaluate(model, X_val, y_val, "Validation (최적 하이퍼파라미터)", label_mode)
    test_auc = evaluate(model, X_test, y_test, "Test (최적 하이퍼파라미터)", label_mode)

    model_path = MODEL_DIR / f"01_{role}_xgboost_bayesian_{label_mode}.json"
    model.save_model(model_path)
    print(f"[saved] {model_path}")

    summary_path = MODEL_DIR / f"01_{role}_xgboost_bayesian_{label_mode}_summary.json"
    summary_path.write_text(
        json.dumps({
            "role": role, "label_mode": label_mode,
            "feature_cols": feature_cols,
            "best_params": p,
            "val_auc": val_auc,
            "test_auc": test_auc,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {summary_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["bullpen", "starter", "both"], default="both")
    parser.add_argument("--label_mode", choices=["3class", "binary", "both"], default="both")
    parser.add_argument("--n_iter", type=int, default=25, help="베이지안 탐색 반복 횟수")
    parser.add_argument("--init_points", type=int, default=8, help="초기 무작위 탐색 횟수")
    args = parser.parse_args()

    roles = ["bullpen", "starter"] if args.role == "both" else [args.role]
    label_modes = ["3class", "binary"] if args.label_mode == "both" else [args.label_mode]
    for role in roles:
        for label_mode in label_modes:
            run(role, label_mode, args.n_iter, args.init_points)


if __name__ == "__main__":
    main()
