"""
항목 2: 같은 부상 사건에서 나온 여러 양성 window를 train에서 어떻게 쓸지 3가지로
비교한다. 01번(XGBoost) 이진분류 모델을 기준으로 진행한다(제일 빠르게 학습되고,
이미 최적 하이퍼파라미터가 나와있어서 그대로 재사용 가능).

방법 A: 모든 양성 window 사용 (현재 방식, 01번과 동일)
방법 B: 부상 사건당 마지막(가장 임박한) 양성 window 1개만 사용
방법 C: 모든 양성 window 사용하되, 사건별 가중치를 나눔(가중치 합이 사건당 1이 되게)

공정한 비교를 위해 세 방법 모두:
  - 입력변수: 01번 summary에 저장된 feature_cols 그대로 재사용(다시 계산 안 함)
  - 모델 하이퍼파라미터: 01번 summary에 저장된 best_params 그대로 재사용
    (베이지안 재탐색 안 함 - 바뀌는 건 학습 데이터의 양성 window 처리 방식뿐)
  - validation/test: 항상 전체 window 그대로 사용(세 모델이 같은 시험지를 풀어야 함)

평가는 episode_eval_utils(항목 3에서 만든 것)를 그대로 재사용해서 window-level +
episode-level 지표를 둘 다 낸다.

주의: il_start_date는 음성(label=0) 행에도 채워져 있다(그 시점 기준 "다음" 부상
사건의 날짜 - horizon 밖이라 라벨만 0일 뿐). 그래서 사건 단위로 묶을 때는 반드시
label==1인 행만 갖고 (player_id, il_start_date)로 묶어야 한다 - 안 그러면 음성
행까지 같은 사건으로 잘못 묶인다.

출력: models/06_window_usage_comparison_xgboost.xlsx
"""
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

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


def xgb_params_from_summary(best_params: dict) -> dict:
    """01_xgboost_bayesian.py의 xgb_base_params(binary)와 동일한 구성.
    01번은 파일명이 숫자로 시작해 import가 안 되므로 그대로 복붙."""
    return dict(
        max_depth=int(round(best_params["max_depth"])), learning_rate=best_params["learning_rate"],
        n_estimators=2000, subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"], min_child_weight=best_params["min_child_weight"],
        reg_lambda=best_params["reg_lambda"], tree_method="hist", enable_categorical=True,
        early_stopping_rounds=50, random_state=42,
        objective="binary:logistic", eval_metric="logloss",
    )


def make_variant_A(train_df: pd.DataFrame):
    """방법 A: 전체 그대로, class-balanced 가중치만."""
    y = train_df["label"].astype(int).to_numpy()
    w = compute_sample_weight("balanced", y)
    return train_df, w


def make_variant_B(train_df: pd.DataFrame):
    """방법 B: 사건(player_id, il_start_date)당 마지막 양성 window 1개만 남김."""
    pos = train_df[train_df["label"] == 1]
    neg = train_df[train_df["label"] == 0]
    last_idx = pos.groupby(["player_id", "il_start_date"])["window_end_date"].idxmax()
    pos_last = pos.loc[last_idx]
    train_b = pd.concat([neg, pos_last]).sort_index()
    y = train_b["label"].astype(int).to_numpy()
    w = compute_sample_weight("balanced", y)
    return train_b, w


def make_variant_C(train_df: pd.DataFrame):
    """방법 C: 전체 사용 + class-balanced 가중치를 사건별 양성 window 수로 나눔.
    (음성 가중치는 그대로, 양성 가중치만 그 사건에 딸린 양성 window 수로 나눠서
    사건 하나의 총 영향력이 class-balanced 가중치 1개분과 같아지게 만든다.)"""
    y = train_df["label"].astype(int).to_numpy()
    w = compute_sample_weight("balanced", y).astype(float)

    pos_mask = (train_df["label"] == 1).to_numpy()
    pos = train_df[train_df["label"] == 1]
    ep_size = pos.groupby(["player_id", "il_start_date"])["label"].transform("size")
    w[pos_mask] = w[pos_mask] / ep_size.to_numpy()
    return train_df, w


VARIANTS = {"A_전체사용": make_variant_A, "B_마지막만": make_variant_B, "C_사건별가중치": make_variant_C}


def build_score_df(df: pd.DataFrame, y_score: np.ndarray) -> pd.DataFrame:
    d = df[["player_id", "il_start_date", "label", "days_to_injury"]].copy()
    d["y_score"] = y_score
    return d


def run_role(role: str):
    print(f"\n===== {role} =====")
    summary = json.loads(
        (MODEL_DIR / f"01_{role}_xgboost_bayesian_binary_summary.json").read_text(encoding="utf-8")
    )
    feature_cols = summary["feature_cols"]
    params = xgb_params_from_summary(summary["best_params"])

    splits = load_role(role, exclude_other=True, binarize=True)
    full_train = splits["train"]
    X_val, y_val = splits["val"][feature_cols], splits["val"]["label"].astype(int)
    X_test, y_test = splits["test"][feature_cols], splits["test"]["label"].astype(int)

    sweep_rows, chosen_rows, window_rows, episode_rows = [], [], [], []

    for variant_name, make_variant in VARIANTS.items():
        train_df, sample_weight = make_variant(full_train)
        print(f"[{variant_name}] train rows={len(train_df):,}  양성={int((train_df['label']==1).sum())}  "
              f"(원본 양성={int((full_train['label']==1).sum())})")

        model = xgb.XGBClassifier(**params)
        model.fit(train_df[feature_cols], train_df["label"].astype(int), sample_weight=sample_weight,
                  eval_set=[(X_val, y_val)], verbose=False)

        val_scores = build_score_df(splits["val"], model.predict_proba(X_val)[:, 1])
        test_scores = build_score_df(splits["test"], model.predict_proba(X_test)[:, 1])

        sweep_val = threshold_sweep(val_scores, THRESHOLD_GRID)
        sweep_val.insert(0, "role", role)
        sweep_val.insert(0, "variant", variant_name)
        sweep_rows.append(sweep_val)

        # threshold와 무관한 전체 window 지표(참고용, PR-AUC/ROC-AUC)
        overall = window_metrics(test_scores["label"].to_numpy(), test_scores["y_score"].to_numpy(), threshold=0.5)
        boot_w = pitcher_cluster_bootstrap_window(test_scores, n_boot=N_BOOT)

        for budget in FALSE_ALERT_BUDGETS:
            picked = pick_threshold_for_budget(sweep_val, budget)
            if picked is None:
                print(f"  [{variant_name}] budget={budget} 만족 threshold 없음")
                continue
            threshold = float(picked["threshold"])

            wm = window_metrics(test_scores["label"].to_numpy(), test_scores["y_score"].to_numpy(), threshold=threshold)
            pos_test = test_scores[test_scores["label"] == 1]
            ep = episode_recall_at_threshold(pos_test, threshold)
            boot_e = pitcher_cluster_bootstrap_episode_recall(test_scores, threshold, n_boot=N_BOOT)

            chosen_rows.append({
                "variant": variant_name, "role": role, "false_alert_budget": budget, "threshold": threshold,
            })
            window_rows.append({
                "variant": variant_name, "role": role, "false_alert_budget": budget, "threshold": threshold,
                "pr_auc": overall["pr_auc"], "pr_auc_ci_low": boot_w["pr_auc_ci"][0], "pr_auc_ci_high": boot_w["pr_auc_ci"][1],
                "roc_auc": overall["roc_auc"], "roc_auc_ci_low": boot_w["roc_auc_ci"][0], "roc_auc_ci_high": boot_w["roc_auc_ci"][1],
                "precision": wm["precision"], "recall": wm["recall"], "f1": wm["f1"], "f2": wm["f2"],
                "tp": wm["tp"], "fp": wm["fp"], "fn": wm["fn"], "tn": wm["tn"],
            })
            episode_rows.append({
                "variant": variant_name, "role": role, "false_alert_budget": budget, "threshold": threshold,
                "n_episodes": ep["n_episodes"], "n_detected": ep["n_detected"],
                "episode_recall": ep["episode_recall"],
                "episode_recall_ci_low": boot_e["episode_recall_ci"][0],
                "episode_recall_ci_high": boot_e["episode_recall_ci"][1],
                "median_lead_time_days": ep["median_lead_time"],
            })
            print(f"  [{variant_name}] budget={budget:.0%} -> threshold={threshold:.3f}  "
                  f"episode_recall={ep['episode_recall']:.3f}  PR-AUC={overall['pr_auc']:.4f}")

    return (pd.concat(sweep_rows, ignore_index=True), pd.DataFrame(chosen_rows),
            pd.DataFrame(window_rows), pd.DataFrame(episode_rows))


def main():
    all_sweep, all_chosen, all_window, all_episode = [], [], [], []
    for role in ("bullpen", "starter"):
        sweep, chosen, window, episode = run_role(role)
        all_sweep.append(sweep)
        all_chosen.append(chosen)
        all_window.append(window)
        all_episode.append(episode)

    out_path = MODEL_DIR / "06_window_usage_comparison_xgboost.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.concat(all_sweep, ignore_index=True).to_excel(writer, sheet_name="validation_threshold_sweep", index=False)
        pd.concat(all_chosen, ignore_index=True).to_excel(writer, sheet_name="선택된_threshold", index=False)
        pd.concat(all_window, ignore_index=True).to_excel(writer, sheet_name="test_window_지표", index=False)
        pd.concat(all_episode, ignore_index=True).to_excel(writer, sheet_name="test_episode_지표", index=False)

    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
