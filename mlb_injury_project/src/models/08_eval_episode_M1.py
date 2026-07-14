"""
07번(M1 변수 포함 XGBoost 이진분류) 모델에 3번 항목 평가(episode_eval_utils)를
그대로 적용해서, M0(05번 결과, xgboost 행)와 window/episode 단위로 비교한다.

출력: models/08_episode_eval_M1.xlsx
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import MODEL_DIR, load_role  # noqa: E402
from episode_eval_utils import (  # noqa: E402
    window_metrics, threshold_sweep, pick_threshold_for_budget, episode_recall_at_threshold,
    pitcher_cluster_bootstrap_window, pitcher_cluster_bootstrap_episode_recall,
)

FALSE_ALERT_BUDGETS = [0.01, 0.05, 0.10]
THRESHOLD_GRID = np.concatenate([
    np.arange(0.01, 0.10, 0.01),
    np.arange(0.10, 0.55, 0.05),
])
N_BOOT = 1000


def build_score_df(df: pd.DataFrame, y_score: np.ndarray) -> pd.DataFrame:
    d = df[["player_id", "il_start_date", "label", "days_to_injury"]].copy()
    d["y_score"] = y_score
    return d


def run_role(role: str):
    print(f"\n===== {role} (M1) =====")
    summary = json.loads(
        (MODEL_DIR / f"07_{role}_xgboost_binary_M1_summary.json").read_text(encoding="utf-8")
    )
    feature_cols = summary["feature_cols"]
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / f"07_{role}_xgboost_binary_M1.json")

    splits = load_role(role, exclude_other=True, binarize=True)
    val_scores = build_score_df(splits["val"], model.predict_proba(splits["val"][feature_cols])[:, 1])
    test_scores = build_score_df(splits["test"], model.predict_proba(splits["test"][feature_cols])[:, 1])

    sweep_val = threshold_sweep(val_scores, THRESHOLD_GRID)
    sweep_val.insert(0, "role", role)
    sweep_val.insert(0, "stage", "M1")

    overall = window_metrics(test_scores["label"].to_numpy(), test_scores["y_score"].to_numpy(), threshold=0.5)
    boot_w = pitcher_cluster_bootstrap_window(test_scores, n_boot=N_BOOT)

    window_rows, episode_rows = [], []
    for budget in FALSE_ALERT_BUDGETS:
        picked = pick_threshold_for_budget(sweep_val, budget)
        if picked is None:
            print(f"  budget={budget} 만족 threshold 없음")
            continue
        threshold = float(picked["threshold"])

        wm = window_metrics(test_scores["label"].to_numpy(), test_scores["y_score"].to_numpy(), threshold=threshold)
        pos_test = test_scores[test_scores["label"] == 1]
        ep = episode_recall_at_threshold(pos_test, threshold)
        boot_e = pitcher_cluster_bootstrap_episode_recall(test_scores, threshold, n_boot=N_BOOT)

        window_rows.append({
            "stage": "M1", "role": role, "false_alert_budget": budget, "threshold": threshold,
            "pr_auc": overall["pr_auc"], "pr_auc_ci_low": boot_w["pr_auc_ci"][0], "pr_auc_ci_high": boot_w["pr_auc_ci"][1],
            "roc_auc": overall["roc_auc"], "roc_auc_ci_low": boot_w["roc_auc_ci"][0], "roc_auc_ci_high": boot_w["roc_auc_ci"][1],
            "precision": wm["precision"], "recall": wm["recall"], "f1": wm["f1"], "f2": wm["f2"],
            "tp": wm["tp"], "fp": wm["fp"], "fn": wm["fn"], "tn": wm["tn"],
        })
        episode_rows.append({
            "stage": "M1", "role": role, "false_alert_budget": budget, "threshold": threshold,
            "n_episodes": ep["n_episodes"], "n_detected": ep["n_detected"],
            "episode_recall": ep["episode_recall"],
            "episode_recall_ci_low": boot_e["episode_recall_ci"][0],
            "episode_recall_ci_high": boot_e["episode_recall_ci"][1],
            "median_lead_time_days": ep["median_lead_time"],
        })
        print(f"  budget={budget:.0%} -> threshold={threshold:.3f}  "
              f"episode_recall={ep['episode_recall']:.3f}  PR-AUC={overall['pr_auc']:.4f}")

    return sweep_val, pd.DataFrame(window_rows), pd.DataFrame(episode_rows)


def main():
    all_sweep, all_window, all_episode = [], [], []
    for role in ("bullpen", "starter"):
        sweep, window, episode = run_role(role)
        all_sweep.append(sweep)
        all_window.append(window)
        all_episode.append(episode)

    m0_path = MODEL_DIR / "05_episode_eval_tree_models.xlsx"
    m0_window = pd.read_excel(m0_path, sheet_name="test_window_지표")
    m0_episode = pd.read_excel(m0_path, sheet_name="test_episode_지표")
    m0_window = m0_window[m0_window["model"] == "xgboost"].copy()
    m0_episode = m0_episode[m0_episode["model"] == "xgboost"].copy()
    m0_window.insert(0, "stage", "M0")
    m0_episode.insert(0, "stage", "M0")

    out_path = MODEL_DIR / "08_episode_eval_M1.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.concat(all_sweep, ignore_index=True).to_excel(writer, sheet_name="validation_threshold_sweep", index=False)
        pd.concat([m0_window, pd.concat(all_window, ignore_index=True)], ignore_index=True).to_excel(
            writer, sheet_name="test_window_M0_vs_M1", index=False)
        pd.concat([m0_episode, pd.concat(all_episode, ignore_index=True)], ignore_index=True).to_excel(
            writer, sheet_name="test_episode_M0_vs_M1", index=False)

    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
