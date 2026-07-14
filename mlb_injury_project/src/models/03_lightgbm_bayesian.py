"""
모델 3 (공식 파이프라인 1단계). LightGBM + Bayesian Optimization, rolling-window
데이터(bullpen/starter_window_dataset.parquet - 01/02번과 완전히 같은 입력/전처리).
'그 외'(label=3) 행은 항상 제외하고, --label_mode로 두 가지 분류 방식을 둘 다
지원한다(01/02번과 동일).
  3class : 0(안다침)/1(어깨)/2(팔꿈치) 3종 분류
  binary : 1과 2를 합쳐서 0(안다침) vs 1(어깨 또는 팔꿈치) 이진분류

01번(XGBoost)과 다른 점은 딱 하나, 모델을 LightGBM으로 바꾼 것뿐이다. 데이터/전처리/
라벨 기준을 동일하게 맞춰야 세 모델 성능을 공정하게 비교할 수 있다.

탐색 대상 하이퍼파라미터 6개와 범위는 LightGBM 공식 문서(Parameters Tuning 가이드)와
여러 튜닝 튜토리얼에서 흔히 쓰이는 값을 참고해 정했다. XGBoost의 max_depth 대신
LightGBM에서 관용적으로 더 중요하게 다루는 num_leaves를 썼다(트리 깊이 대신 리프
개수로 복잡도를 조절하는 게 LightGBM의 leaf-wise 성장 방식에 더 맞음).
  num_leaves       (16, 256)  : 리프 개수. 많을수록 복잡한 트리(과적합 위험).
  learning_rate    (0.01, 0.3): 각 트리 반영 비율. 작을수록 안정적이지만 느림.
  feature_fraction (0.5, 1.0) : 트리마다 샘플링할 컬럼 비율(XGBoost의 colsample_bytree).
  bagging_fraction (0.5, 1.0) : 트리마다 샘플링할 행 비율(XGBoost의 subsample).
  min_child_samples(5, 100)   : 리프 노드가 가지는 최소 샘플 수. 클수록 보수적.
  reg_lambda       (0.5, 5.0) : L2 정규화 강도.
(bagging_fraction이 실제로 적용되려면 bagging_freq>0이 필요해서 1로 고정한다.)

입력변수 정리: 01/02번과 동일하게 상관계수 0.9 초과 컬럼을 select_uncorrelated_features로
미리 제거한다(트리 기반 모델이라 PCA는 적용하지 않음 - 01번과 동일한 판단).

실행: python 03_lightgbm_bayesian.py --role both --label_mode 3class
      python 03_lightgbm_bayesian.py --role both --label_mode binary
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import CATEGORICAL_COLS, MODEL_DIR, load_role, numeric_feature_cols  # noqa: E402
from feature_selection import select_uncorrelated_features  # noqa: E402

CORR_THRESHOLD = 0.9

PBOUNDS = {
    "num_leaves": (16, 256),
    "learning_rate": (0.01, 0.3),
    "feature_fraction": (0.5, 1.0),
    "bagging_fraction": (0.5, 1.0),
    "min_child_samples": (5, 100),
    "reg_lambda": (0.5, 5.0),
}


def lgb_base_params(label_mode: str, num_leaves, learning_rate, feature_fraction,
                     bagging_fraction, min_child_samples, reg_lambda):
    common = dict(
        num_leaves=int(round(num_leaves)), learning_rate=learning_rate, n_estimators=2000,
        feature_fraction=feature_fraction, bagging_fraction=bagging_fraction, bagging_freq=1,
        min_child_samples=int(round(min_child_samples)), reg_lambda=reg_lambda,
        random_state=42, verbose=-1,
    )
    if label_mode == "binary":
        return dict(objective="binary", metric="binary_logloss", **common)
    return dict(objective="multiclass", num_class=3, metric="multi_logloss", **common)


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

    def objective(num_leaves, learning_rate, feature_fraction, bagging_fraction,
                  min_child_samples, reg_lambda):
        params = lgb_base_params(label_mode, num_leaves, learning_rate, feature_fraction,
                                  bagging_fraction, min_child_samples, reg_lambda)
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
        proba = model.predict_proba(X_val)
        try:
            return auc_score(y_val, proba, label_mode)
        except ValueError:
            return 0.0

    return objective


def run(role: str, label_mode: str, n_iter: int, init_points: int):
    print(f"\n{'=' * 70}\n[모델3:{label_mode}] LightGBM + Bayesian Optimization - {role.upper()}\n{'=' * 70}")
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

    trials_path = MODEL_DIR / f"03_{role}_lightgbm_bayesian_{label_mode}_trials.json"
    trials_path.write_text(
        json.dumps([{"target": r["target"], "params": r["params"]} for r in optimizer.res],
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {trials_path}")

    p = best["params"]
    final_params = lgb_base_params(label_mode, p["num_leaves"], p["learning_rate"], p["feature_fraction"],
                                    p["bagging_fraction"], p["min_child_samples"], p["reg_lambda"])
    X_train, y_train = xy["train"]
    X_val, y_val = xy["val"]
    X_test, y_test = xy["test"]
    sample_weight = compute_sample_weight("balanced", y_train)

    model = lgb.LGBMClassifier(**final_params)
    model.fit(X_train, y_train, sample_weight=sample_weight,
              eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    print(f"best_iteration={model.best_iteration_}")

    val_auc = evaluate(model, X_val, y_val, "Validation (최적 하이퍼파라미터)", label_mode)
    test_auc = evaluate(model, X_test, y_test, "Test (최적 하이퍼파라미터)", label_mode)

    # LightGBM의 booster_.save_model()은 내부적으로 C API의 fopen을 써서 경로에
    # 한글 등 비-ASCII 문자가 있으면 "not available for writes" 오류가 난다(이
    # 프로젝트 경로 자체에 "세종대", "학부연구생1"이 들어있어 실제로 발생함).
    # model_to_string()으로 모델을 문자열로 뽑아 파이썬 자체 파일 IO로 저장해 우회한다.
    model_path = MODEL_DIR / f"03_{role}_lightgbm_bayesian_{label_mode}.txt"
    model_path.write_text(model.booster_.model_to_string(), encoding="utf-8")
    print(f"[saved] {model_path}")

    summary_path = MODEL_DIR / f"03_{role}_lightgbm_bayesian_{label_mode}_summary.json"
    summary_path.write_text(
        json.dumps({
            "role": role, "label_mode": label_mode,
            "feature_cols": feature_cols,
            "best_params": p,
            "best_iteration": model.best_iteration_,
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
