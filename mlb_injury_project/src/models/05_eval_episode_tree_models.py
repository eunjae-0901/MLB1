"""
모델 3번 항목(window 단위 vs episode 단위 평가)을 01번(XGBoost)/03번(LightGBM)
이진분류(binary) 모델에 적용한다. 둘 다 GPU 없이 로컬에서 바로 돌아간다
(02/04번 DNN은 GPU 환경에서 06_eval_episode_dnn_models.py로 따로 평가).

절차:
  1) validation에서 threshold를 여러 개 훑어(threshold_sweep) episode recall과
     false alert rate(음성 window 중 잘못 경고한 비율)의 관계를 표로 만든다.
  2) false alert rate 예산 1%/5%/10%마다 그 예산을 넘지 않는 선에서 episode recall이
     가장 높은 threshold를 고른다(validation만 보고 결정 - test는 안 봄).
  3) 그 threshold들을 test에 그대로 적용해서 window 단위 지표(PR-AUC/ROC-AUC/precision/
     recall/F1/F2/confusion matrix)와 episode 단위 지표(episode recall, lead time)를
     계산한다.
  4) test 지표의 95% 신뢰구간은 투수 단위 cluster bootstrap으로 계산한다(window을
     독립으로 취급하면 신뢰구간이 지나치게 좁게 나옴).

출력: models/05_episode_eval_tree_models.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import CATEGORICAL_COLS, MODEL_DIR, load_role, numeric_feature_cols  # noqa: E402
from feature_selection import select_uncorrelated_features  # noqa: E402
from episode_eval_utils import (  # noqa: E402
    window_metrics, threshold_sweep, pick_threshold_for_budget, episode_recall_at_threshold,
    pitcher_cluster_bootstrap_window, pitcher_cluster_bootstrap_episode_recall,
)

CORR_THRESHOLD = 0.9
FALSE_ALERT_BUDGETS = [0.01, 0.05, 0.10]
THRESHOLD_GRID = np.concatenate([
    np.arange(0.01, 0.10, 0.01),
    np.arange(0.10, 0.55, 0.05),
])
N_BOOT = 1000


def load_xgboost(role: str):
    splits = load_role(role, exclude_other=True, binarize=True)
    all_num_cols = numeric_feature_cols(splits["train"])
    kept_num_cols = select_uncorrelated_features(splits["train"], all_num_cols, threshold=CORR_THRESHOLD)
    feature_cols = kept_num_cols + CATEGORICAL_COLS

    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / f"01_{role}_xgboost_bayesian_binary.json")

    def predict(df):
        return model.predict_proba(df[feature_cols])[:, 1]

    return splits, predict


def load_lightgbm(role: str):
    splits = load_role(role, exclude_other=True, binarize=True)
    all_num_cols = numeric_feature_cols(splits["train"])
    kept_num_cols = select_uncorrelated_features(splits["train"], all_num_cols, threshold=CORR_THRESHOLD)
    feature_cols = kept_num_cols + CATEGORICAL_COLS

    model_path = MODEL_DIR / f"03_{role}_lightgbm_bayesian_binary.txt"
    booster = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))

    def predict(df):
        return booster.predict(df[feature_cols])

    return splits, predict


LOADERS = {"xgboost": load_xgboost, "lightgbm": load_lightgbm}


def build_score_df(splits: dict, predict) -> dict:
    """각 split에 y_score 컬럼을 추가한 평가용 데이터프레임을 만든다."""
    out = {}
    for s, df in splits.items():
        d = df[["player_id", "il_start_date", "label", "days_to_injury"]].copy()
        d["y_score"] = predict(df)
        out[s] = d
    return out


def evaluate_model(model_name: str, role: str):
    print(f"[{model_name}/{role}] 로딩 및 예측 중...")
    splits, predict = LOADERS[model_name](role)
    scored = build_score_df(splits, predict)

    val_df, test_df = scored["val"], scored["test"]

    sweep_val = threshold_sweep(val_df, THRESHOLD_GRID)
    sweep_val.insert(0, "role", role)
    sweep_val.insert(0, "model", model_name)

    chosen_rows = []
    test_window_rows = []
    test_episode_rows = []
    for budget in FALSE_ALERT_BUDGETS:
        picked = pick_threshold_for_budget(sweep_val, budget)
        if picked is None:
            print(f"  [경고] false_alert budget={budget} 만족하는 threshold 없음 (모든 threshold가 그보다 높은 FA율)")
            continue
        threshold = float(picked["threshold"])

        wm = window_metrics(test_df["label"].to_numpy(), test_df["y_score"].to_numpy(), threshold=threshold)
        boot_w = pitcher_cluster_bootstrap_window(test_df, n_boot=N_BOOT)

        pos_test = test_df[test_df["label"] == 1]
        ep = episode_recall_at_threshold(pos_test, threshold)
        boot_e = pitcher_cluster_bootstrap_episode_recall(test_df, threshold, n_boot=N_BOOT)

        chosen_rows.append({
            "model": model_name, "role": role, "false_alert_budget": budget,
            "chosen_threshold": threshold,
            "val_episode_recall_at_pick": picked["episode_recall"],
            "val_false_alert_rate_at_pick": picked["false_alert_rate"],
        })
        test_window_rows.append({
            "model": model_name, "role": role, "false_alert_budget": budget,
            "threshold": threshold,
            "pr_auc": wm["pr_auc"], "pr_auc_ci_low": boot_w["pr_auc_ci"][0], "pr_auc_ci_high": boot_w["pr_auc_ci"][1],
            "roc_auc": wm["roc_auc"], "roc_auc_ci_low": boot_w["roc_auc_ci"][0], "roc_auc_ci_high": boot_w["roc_auc_ci"][1],
            "precision": wm["precision"], "recall": wm["recall"], "f1": wm["f1"], "f2": wm["f2"],
            "tp": wm["tp"], "fp": wm["fp"], "fn": wm["fn"], "tn": wm["tn"],
        })
        test_episode_rows.append({
            "model": model_name, "role": role, "false_alert_budget": budget,
            "threshold": threshold,
            "n_episodes": ep["n_episodes"], "n_detected": ep["n_detected"],
            "episode_recall": ep["episode_recall"],
            "episode_recall_ci_low": boot_e["episode_recall_ci"][0],
            "episode_recall_ci_high": boot_e["episode_recall_ci"][1],
            "median_lead_time_days": ep["median_lead_time"],
        })
        print(f"  budget={budget:.0%} -> threshold={threshold:.3f}  "
              f"test episode_recall={ep['episode_recall']:.3f}  test PR-AUC={wm['pr_auc']:.4f}")

    return sweep_val, pd.DataFrame(chosen_rows), pd.DataFrame(test_window_rows), pd.DataFrame(test_episode_rows)


def main():
    all_sweep, all_chosen, all_window, all_episode = [], [], [], []
    for model_name in ("xgboost", "lightgbm"):
        for role in ("bullpen", "starter"):
            sweep, chosen, window, episode = evaluate_model(model_name, role)
            all_sweep.append(sweep)
            all_chosen.append(chosen)
            all_window.append(window)
            all_episode.append(episode)

    out_path = MODEL_DIR / "05_episode_eval_tree_models.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.concat(all_sweep, ignore_index=True).to_excel(writer, sheet_name="validation_threshold_sweep", index=False)
        pd.concat(all_chosen, ignore_index=True).to_excel(writer, sheet_name="선택된_threshold", index=False)
        pd.concat(all_window, ignore_index=True).to_excel(writer, sheet_name="test_window_지표", index=False)
        pd.concat(all_episode, ignore_index=True).to_excel(writer, sheet_name="test_episode_지표", index=False)

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
