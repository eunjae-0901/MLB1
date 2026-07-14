"""
02_dnn_bayesian.py로 학습된 모델(3class/binary 둘 다)을 다시 불러와서 결과를 엑셀
파일 하나에 정리한다. 01_export_xgboost_results.py(XGBoost)와 시트 구성을 동일하게
맞춰서 두 모델을 나란히 비교하기 쉽게 했다.
*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 *** (GPU 없이 CPU로도 돌아가지만
torch 자체는 설치되어 있어야 한다)

시트 구성은 01_export_xgboost_results.py와 동일: 하이퍼파라미터/성능지표/탐색기록/
예측_{역할}_{3종분류|이진분류}.

출력: models/02_dnn_bayesian_results.xlsx
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import MODEL_DIR  # noqa: E402
from dnn_bayesian_utils import DEVICE, TabularMLP, prepare_data  # noqa: E402

LABEL_NAMES_3CLASS = {0: "안다침", 1: "어깨", 2: "팔꿈치"}
LABEL_NAMES_BINARY = {0: "안다침", 1: "어깨또는팔꿈치"}
LABEL_MODE_KOR = {"3class": "3종분류", "binary": "이진분류"}


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


def predict_split(model, loader):
    model.eval()
    all_pred, all_proba = [], []
    with torch.no_grad():
        for x_num, x_cat, _y in loader:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            proba = torch.softmax(model(x_num, x_cat), dim=1).cpu().numpy()
            all_proba.append(proba)
            all_pred.append(proba.argmax(axis=1))
    return np.concatenate(all_pred), np.concatenate(all_proba)


def main():
    hyperparam_rows = []
    metrics_rows = []
    search_frames = []
    prediction_frames = {}

    for role in ("bullpen", "starter"):
        for label_mode in ("3class", "binary"):
            summary = json.loads(
                (MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}_summary.json").read_text(encoding="utf-8")
            )
            datasets, meta, _y_train, _splits = prepare_data(role, label_mode)
            label_names = LABEL_NAMES_BINARY if label_mode == "binary" else LABEL_NAMES_3CLASS

            model = TabularMLP(
                num_dim=meta["num_dim"], p_throws_vocab=meta["p_throws_vocab_size"],
                country_vocab=meta["country_vocab_size"], hidden_sizes=summary["hidden_sizes"],
                num_classes=summary["n_classes"], dropout=summary["best_params"]["dropout"],
            ).to(DEVICE)
            model.load_state_dict(
                torch.load(MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}.pt", map_location=DEVICE)
            )

            pred_rows = []
            role_metrics = {}
            for split in ("train", "val", "test"):
                loader = DataLoader(datasets[split], batch_size=512)
                pred, proba = predict_split(model, loader)
                y_true = datasets[split].y.numpy()

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

            p = summary["best_params"]
            hyperparam_rows.append({
                "role": role, "label_mode": LABEL_MODE_KOR[label_mode],
                "learning_rate": round(10 ** p["learning_rate"], 6),
                "hidden_layer_init": round(p["hidden_layer_init"], 3),
                "hidden_node_init": round(p["hidden_node_init"], 3),
                "batch_size": int(p["batch_size"]), "dropout": round(p["dropout"], 3),
                "hidden_sizes": str(summary["hidden_sizes"]), "pca_dim": summary["pca_dim"],
                "train_auc": round(role_metrics["train"]["roc_auc"], 4),
                "val_auc": round(role_metrics["val"]["roc_auc"], 4),
                "test_auc": round(role_metrics["test"]["roc_auc"], 4),
            })

            trials = json.loads(
                (MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}_trials.json").read_text(encoding="utf-8")
            )
            search_df = pd.json_normalize(
                [{"role": role, "label_mode": LABEL_MODE_KOR[label_mode], "val_auc": t["target"], **t["params"]}
                 for t in trials]
            )
            search_df = search_df.sort_values("val_auc", ascending=False).reset_index(drop=True)
            search_df.insert(0, "순위", range(1, len(search_df) + 1))
            search_frames.append(search_df)

    out_path = MODEL_DIR / "02_dnn_bayesian_results.xlsx"
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
