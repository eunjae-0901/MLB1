"""시나리오 1: train만 case-control로 재구성한다.

Validation과 test는 실제 MLB의 자연 부상 비율을 유지한다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SEED = 42
SCENARIO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCENARIO_DIR.parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = DATA_DIR / "case_control"
ROLES = ("bullpen", "starter")
RATIOS = (3, 5)
CAP_RATIO = 3
CAP_PER_PLAYER = 3


def load_base(role: str) -> pd.DataFrame:
    """기타 부상(label=3)을 제외하고 이진분류용 원본을 반환한다."""
    path = DATA_DIR / f"{role}_dataset.csv"
    df = pd.read_csv(path)
    before = len(df)
    df = df.loc[df["label"] != 3].copy()
    df["label"] = (df["label"] > 0).astype("int8")
    print(f"[{role}] 기타 부상 제외: {before:,} -> {len(df):,}행")
    return df


def sample_negatives(
    negatives: pd.DataFrame,
    n_target: int,
    cap_per_player: int | None = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """정상 window를 무작위 추출하되 필요하면 선수별 후보 수를 제한한다."""
    pool = negatives
    if cap_per_player is not None:
        pool = (
            negatives.sample(frac=1.0, random_state=seed)
            .groupby("player_id", sort=False, group_keys=False)
            .head(cap_per_player)
        )
    return pool.sample(n=min(n_target, len(pool)), random_state=seed)


def build_case_control_train(
    df: pd.DataFrame,
    ratio: int,
    cap_per_player: int | None = None,
) -> pd.DataFrame:
    """train만 1:ratio로 만들고 validation/test는 원본 그대로 붙인다."""
    train = df.loc[df["split"] == "train"]
    positive = train.loc[train["label"] == 1]
    negative = train.loc[train["label"] == 0]
    sampled_negative = sample_negatives(
        negative,
        n_target=len(positive) * ratio,
        cap_per_player=cap_per_player,
    )
    sampled_train = pd.concat([positive, sampled_negative], ignore_index=True)
    sampled_train = sampled_train.sample(frac=1.0, random_state=SEED)
    untouched_eval = df.loc[df["split"].isin(["val", "test"])]
    out = pd.concat([sampled_train, untouched_eval], ignore_index=True)

    cap_text = f", 선수당 정상 후보 최대 {cap_per_player}개" if cap_per_player else ""
    print(
        f"  train 1:{ratio}{cap_text}: 양성 {len(positive):,}, "
        f"음성 {len(sampled_negative):,}"
    )
    return out


def distribution_rows(role: str, variant: str, df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        part = df.loc[df["split"] == split]
        rows.append(
            {
                "role": role,
                "variant": variant,
                "split": split,
                "n_rows": len(part),
                "n_positive": int(part["label"].sum()),
                "positive_rate": float(part["label"].mean()),
                "n_players": int(part["player_id"].nunique()),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for role in ROLES:
        base = load_base(role)
        summary.extend(distribution_rows(role, "baseline", base))
        variants = [(f"cc{ratio}", ratio, None) for ratio in RATIOS]
        variants.append((f"cc{CAP_RATIO}_capped{CAP_PER_PLAYER}", CAP_RATIO, CAP_PER_PLAYER))
        for variant, ratio, cap in variants:
            output = build_case_control_train(base, ratio, cap)
            path = OUT_DIR / f"{role}_dataset_{variant}.csv"
            output.to_csv(path, index=False)
            summary.extend(distribution_rows(role, variant, output))
            print(f"  저장: {path}")

    summary_df = pd.DataFrame(summary)
    summary_path = OUT_DIR / "scenario1_sampling_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print("\n" + summary_df.to_string(index=False))
    print(f"\n요약 저장: {summary_path}")


if __name__ == "__main__":
    main()
