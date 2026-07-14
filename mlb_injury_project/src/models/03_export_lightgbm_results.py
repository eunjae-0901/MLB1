"""
03_lightgbm_bayesian.py로 학습된 모델(3class/binary 둘 다)을 다시 불러와서 결과를
엑셀 파일 하나에 정리한다. 01_export_xgboost_results.py(XGBoost), 02_export_dnn_
results.py(DNN)와 시트 구성을 동일하게 맞춰서 세 모델을 나란히 비교하기 쉽게 했다.

시트 구성은 다른 두 export 스크립트와 동일: 하이퍼파라미터/성능지표/탐색기록/
예측_{역할}_{3종분류|이진분류}.

모델 로딩 시 lgb.Booster(model_file=...) 대신 model_str로 읽는 이유: LightGBM의
파일 로딩도 저장과 마찬가지로 C API의 fopen을 쓰는데, 이 프로젝트 경로에 한글이
포함돼 있어서 그대로 두면 파일을 못 여는 문제가 있다. 파이썬 자체 파일 IO로 문자열을
읽어서 model_str로 넘기면 우회된다.

출력: models/03_lightgbm_bayesian_results.xlsx
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import CATEGORICAL_COLS, MODEL_DIR, load_role, numeric_feature_cols  # noqa: E402
from feature_selection import select_uncorrelated_features  # noqa: E402

CORR_THRESHOLD = 0.9
LABEL_NAMES_3CLASS = {0: "안다침", 1: "어깨", 2: "팔꿈치"}
LABEL_NAMES_BINARY = {0: "안다침", 1: "어깨또는팔꿈치"}
LABEL_MODE_KOR = {"3class": "3종분류", "binary": "이진분류"}


def load_model_and_data(role: str, label_mode: str):
    summary = json.loads(
        (MODEL_DIR / f"03_{role}_lightgbm_bayesian_{label_mode}_summary.json").read_text(encoding="utf-8")
    )
    splits = load_role(role, exclude_other=True, binarize=(label_mode == "binary"))
    all_num_cols = numeric_feature_cols(splits["train"])
    kept_num_cols = select_uncorrelated_features(splits["train"], all_num_cols, threshold=CORR_THRESHOLD)
    feature_cols = kept_num_cols + CATEGORICAL_COLS

    model_path = MODEL_DIR / f"03_{role}_lightgbm_bayesian_{label_mode}.txt"
    booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))

    return booster, splits, feature_cols, summary


def predict_proba(booster, X, label_mode: str):
    raw = booster.predict(X)
    if label_mode == "binary":
        return np.column_stack([1 - raw, raw])
    return raw


def compute_metrics(y_true, y_pred, y_proba, label_mode: str):
    if label_mode == "binary":
        return dict(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_proba[:, 1]),
        )
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, average="macro", zero_division=0),
        recall=recall_score(y_true, y_pred, average="macro", zero_division=0),
        f1_score=f1_score(y_true, y_pred, average="macro", zero_division=0),
        roc_auc=roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr"),
    )


def main():
    hyperparam_rows = []
    metrics_rows = []
    search_frames = []
    prediction_frames = {}

    for role in ("bullpen", "starter"):
        for label_mode in ("3class", "binary"):
            booster, splits, feature_cols, summary = load_model_and_data(role, label_mode)
            p = summary["best_params"]
            label_names = LABEL_NAMES_BINARY if label_mode == "binary" else LABEL_NAMES_3CLASS

            pred_rows = []
            role_metrics = {}
            for split in ("train", "val", "test"):
                df = splits[split]
                X = df[feature_cols]
                y_true = df["label"].astype(int).to_numpy()
                proba = predict_proba(booster, X, label_mode)
                pred = proba.argmax(axis=1)

                role_metrics[split] = compute_metrics(y_true, pred, proba, label_mode)
                metrics_rows.append({
                    "role": role, "label_mode": LABEL_MODE_KOR[label_mode], "dataset": split,
                    **role_metrics[split],
                })

                pred_rows.append(pd.DataFrame({
                    "split": split,
                    "정답": y_true, "정답_분류": [label_names[v] for v in y_true],
                    "예측": pred, "예측_분류": [label_names[v] for v in pred],
                }))
            prediction_frames[f"{role}_{label_mode}"] = pd.concat(pred_rows, ignore_index=True)

            hyperparam_rows.append({
                "role": role, "label_mode": LABEL_MODE_KOR[label_mode],
                "num_leaves": round(p["num_leaves"], 3), "learning_rate": round(p["learning_rate"], 5),
                "feature_fraction": round(p["feature_fraction"], 3),
                "bagging_fraction": round(p["bagging_fraction"], 3),
                "min_child_samples": round(p["min_child_samples"], 3),
                "reg_lambda": round(p["reg_lambda"], 3),
                "n_features_used": len(feature_cols),
                "train_auc": round(role_metrics["train"]["roc_auc"], 4),
                "val_auc": round(role_metrics["val"]["roc_auc"], 4),
                "test_auc": round(role_metrics["test"]["roc_auc"], 4),
            })

            trials = json.loads(
                (MODEL_DIR / f"03_{role}_lightgbm_bayesian_{label_mode}_trials.json").read_text(encoding="utf-8")
            )
            search_df = pd.json_normalize(
                [{"role": role, "label_mode": LABEL_MODE_KOR[label_mode], "val_auc": t["target"], **t["params"]}
                 for t in trials]
            )
            search_df = search_df.sort_values("val_auc", ascending=False).reset_index(drop=True)
            search_df.insert(0, "순위", range(1, len(search_df) + 1))
            search_frames.append(search_df)

    out_path = MODEL_DIR / "03_lightgbm_bayesian_results.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame(hyperparam_rows).to_excel(writer, sheet_name="하이퍼파라미터", index=False)
        pd.DataFrame(metrics_rows).to_excel(writer, sheet_name="성능지표", index=False)
        pd.concat(search_frames, ignore_index=True).to_excel(writer, sheet_name="탐색기록", index=False)
        prediction_frames["bullpen_3class"].to_excel(writer, sheet_name="예측_불펜_3종분류", index=False)
        prediction_frames["bullpen_binary"].to_excel(writer, sheet_name="예측_불펜_이진분류", index=False)
        prediction_frames["starter_3class"].to_excel(writer, sheet_name="예측_선발_3종분류", index=False)
        prediction_frames["starter_binary"].to_excel(writer, sheet_name="예측_선발_이진분류", index=False)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
