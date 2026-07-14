"""
window 단위 예측을 episode(부상 사건) 단위로도 평가하기 위한 공용 유틸.
01/02/03/04번 모델 어디에도 의존하지 않는다 - y_true/y_score/player_id/il_start_date/
days_to_injury만 있으면 되고, 이 값들은 예측 후 데이터프레임에서 그대로 뽑아 쓴다.

용어:
  window-level : 매 등판 시점(window) 하나하나를 독립적인 예측 기회로 보고 평가
  episode-level: 같은 부상 사건에 딸린 여러 양성 window 중 하나라도 threshold를
                 넘기면 그 부상 사건 자체를 "사전 경고 성공"으로 봄

주의: 아래 함수들은 전부 binary label_mode(0=안다침, 1=부상) 기준이다. 3class는
"부상 확률"을 어떻게 정의할지(어깨/팔꿈치 중 하나라도 vs 각각 따로) 애매해서, 우선
binary로만 episode-level 평가를 한다 - 원래 이 평가 방식 자체가 "경고를 울릴지 말지"
이진 결정 문제라 binary가 자연스럽다.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score,
)


def window_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """window 단위 지표. threshold 기준 precision/recall/F1/F2/confusion matrix +
    threshold와 무관한 PR-AUC/ROC-AUC."""
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        pr_auc=average_precision_score(y_true, y_score),
        roc_auc=roc_auc_score(y_true, y_score),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        f2=fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )


def build_episode_table(df: pd.DataFrame) -> pd.DataFrame:
    """label==1(양성) window들을 (player_id, il_start_date) 기준으로 묶어서 부상
    사건 하나당 한 행으로 요약한다. episode_proba = 그 사건에 딸린 모든 양성
    window의 y_score 중 최댓값(하나라도 강하게 의심되면 그 사건은 의심된 것으로 봄)."""
    pos = df[df["label"] == 1].copy()
    episodes = (
        pos.groupby(["player_id", "il_start_date"])
        .agg(episode_proba=("y_score", "max"), n_windows=("y_score", "size"))
        .reset_index()
    )
    return episodes, pos


def _lead_time_for_episode(pos_rows: pd.DataFrame, threshold: float):
    """threshold를 넘긴 window 중 days_to_injury가 가장 큰(=제일 먼저 경고한) 값을
    lead time으로 반환. 하나도 못 넘겼으면 None(탐지 실패)."""
    hit = pos_rows[pos_rows["y_score"] >= threshold]
    if len(hit) == 0:
        return None
    return hit["days_to_injury"].max()


def episode_recall_at_threshold(pos: pd.DataFrame, threshold: float) -> dict:
    """pos: build_episode_table이 반환한 label==1 원본 행(episode 아님, window 단위).
    각 episode(player_id, il_start_date)별로 첫 경고 lead time을 구해 recall/중앙값을 낸다."""
    lead_times = []
    n_episodes = 0
    n_detected = 0
    for (_pid, _il), grp in pos.groupby(["player_id", "il_start_date"]):
        n_episodes += 1
        lt = _lead_time_for_episode(grp, threshold)
        if lt is not None:
            n_detected += 1
            lead_times.append(lt)
    recall = n_detected / n_episodes if n_episodes else float("nan")
    median_lead = float(np.median(lead_times)) if lead_times else float("nan")
    return dict(n_episodes=n_episodes, n_detected=n_detected, episode_recall=recall,
                median_lead_time=median_lead)


def false_alert_rate_at_threshold(neg_scores: np.ndarray, threshold: float) -> float:
    """label==0(음성) window 중 threshold를 넘겨 잘못 경고한 비율."""
    if len(neg_scores) == 0:
        return float("nan")
    return float((neg_scores >= threshold).mean())


def threshold_sweep(df: pd.DataFrame, thresholds) -> pd.DataFrame:
    """df: 한 split(보통 validation)의 window 단위 예측 결과
    (player_id, il_start_date, label, days_to_injury, y_score 컬럼 필요).
    각 threshold마다 episode recall / false alert rate / median lead time을 계산."""
    pos = df[df["label"] == 1]
    neg_scores = df.loc[df["label"] == 0, "y_score"].to_numpy()

    rows = []
    for t in thresholds:
        ep = episode_recall_at_threshold(pos, t)
        fa = false_alert_rate_at_threshold(neg_scores, t)
        rows.append({"threshold": t, "false_alert_rate": fa, **ep})
    return pd.DataFrame(rows)


def pick_threshold_for_budget(sweep_df: pd.DataFrame, budget: float):
    """false_alert_rate <= budget인 threshold 중 가장 작은(=recall이 제일 높은) 값을 고른다.
    (threshold가 클수록 false_alert_rate와 recall 둘 다 단조 감소하므로, budget을 만족하는
    가장 작은 threshold가 그 budget 안에서 recall을 최대화한다.)"""
    ok = sweep_df[sweep_df["false_alert_rate"] <= budget]
    if len(ok) == 0:
        return None
    return ok.sort_values("threshold").iloc[0]


def pitcher_cluster_bootstrap_window(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    """window-level PR-AUC/ROC-AUC의 95% CI를 투수 단위 cluster bootstrap으로 계산.
    같은 투수의 window들은 서로 닮아있어서(부상 여부와 무관하게) 독립으로 보면 신뢰구간이
    지나치게 좁게 나온다 - 그래서 투수를 통째로 복원추출한다."""
    rng = np.random.default_rng(seed)
    pitcher_ids = df["player_id"].unique()
    groups = {pid: g for pid, g in df.groupby("player_id")}

    pr_aucs, roc_aucs = [], []
    for _ in range(n_boot):
        sampled = rng.choice(pitcher_ids, size=len(pitcher_ids), replace=True)
        parts = [groups[pid] for pid in sampled]
        boot_df = pd.concat(parts, ignore_index=True)
        if boot_df["label"].nunique() < 2:
            continue
        pr_aucs.append(average_precision_score(boot_df["label"], boot_df["y_score"]))
        roc_aucs.append(roc_auc_score(boot_df["label"], boot_df["y_score"]))

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {
        "pr_auc_ci": ci(pr_aucs), "pr_auc_mean": float(np.mean(pr_aucs)),
        "roc_auc_ci": ci(roc_aucs), "roc_auc_mean": float(np.mean(roc_aucs)),
        "n_boot_valid": len(pr_aucs),
    }


def pitcher_cluster_bootstrap_episode_recall(df: pd.DataFrame, threshold: float,
                                              n_boot: int = 1000, seed: int = 42) -> dict:
    """episode recall의 95% CI도 투수 단위로 재표집(같은 투수가 여러 부상 사건을 가질
    수 있으므로 이게 제일 보수적)."""
    rng = np.random.default_rng(seed)
    pos = df[df["label"] == 1]
    pitcher_ids = pos["player_id"].unique()
    groups = {pid: g for pid, g in pos.groupby("player_id")}

    recalls = []
    for _ in range(n_boot):
        sampled = rng.choice(pitcher_ids, size=len(pitcher_ids), replace=True)
        parts = [groups[pid] for pid in sampled if pid in groups]
        if not parts:
            continue
        boot_pos = pd.concat(parts, ignore_index=True)
        ep = episode_recall_at_threshold(boot_pos, threshold)
        if ep["n_episodes"] > 0:
            recalls.append(ep["episode_recall"])

    return {
        "episode_recall_ci": (float(np.percentile(recalls, 2.5)), float(np.percentile(recalls, 97.5))),
        "episode_recall_mean": float(np.mean(recalls)),
        "n_boot_valid": len(recalls),
    }
