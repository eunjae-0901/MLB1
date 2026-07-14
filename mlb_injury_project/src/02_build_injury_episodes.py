"""
2~3단계 재현: MLB transaction -> 투수 최초 IL 등재 후보 -> 중복 통합 -> injury episode
-> 어깨/팔꿈치(Strict/Broad) 부상 유형 분류.

입력: data/raw/transactions/transactions_{year}.parquet (2016~2025)
출력: data/interim/injury_episodes/injury_episodes.parquet
      data/interim/injury_episodes/review_needed.xlsx
        (both_strict_flag / both_broad_flag: 어깨+팔꿈치 키워드 동시 매칭
         ambiguous_muscle_flag: biceps/triceps/deltoid/median nerve 등 애매한 근육/신경 언급)
"""
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "transactions"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "injury_episodes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR, END_YEAR = 2016, 2025

PITCHER_POSITIONS = {"RHP", "LHP", "P", "SHP", "TWP"}

EXCLUDE_PATTERN = re.compile(
    r"reinstated|transferred|activated|returned|optioned|designated", re.IGNORECASE
)
PLACED_PATTERN = re.compile(r"\bplaced\b", re.IGNORECASE)
IL_MENTION_PATTERN = re.compile(r"injured list|disabled list", re.IGNORECASE)
POSITION_EXTRACT_PATTERN = re.compile(r"\bplaced\s+([A-Za-z]{1,4})\s+", re.IGNORECASE)

SHOULDER_STRICT_KEYWORDS = [
    r"shoulder", r"rotator cuff", r"scapula", r"scapular",
    r"acromioclavicular", r"ac joint",
]
# labrum/labral은 hip과 함께 나오면 제외 (hip labrum 오분류 방지)
LABRUM_PATTERN = re.compile(r"labrum|labral", re.IGNORECASE)
HIP_PATTERN = re.compile(r"\bhip\b", re.IGNORECASE)

ELBOW_STRICT_KEYWORDS = [
    r"elbow", r"ulnar col{1,2}ateral ligament", r"\bucl\b",
    r"(?:ulnar|unar) nerve", r"tommy john", r"\btjs\b",
    # epicondyle/epicondylitis: 팔꿈치 안쪽/바깥쪽 뼈 돌기 부상(예: Jarrod Parker
    # "fractured right medial epicondyle", Alex Colome "lateral epicondylitis")이
    # 기존 키워드에 안 걸려 "그 외"로 잘못 분류되던 것을 발견해 추가.
    r"epicondyl",
]
ELBOW_BROAD_EXTRA_KEYWORDS = [r"forearm", r"flexor", r"pronator"]

# 어깨/팔꿈치 키워드로 명확히 안 잡히지만 해부학적으로 애매하게 걸쳐있는 근육/신경.
# biceps(이두근)는 장두건 기시부가 어깨 관절 쪽, triceps(삼두근)는 힘줄이 팔꿈치(주두)
# 쪽이라 각각 어깨/팔꿈치 부상군으로 볼 여지가 있다. 자동으로 어느 한쪽으로 단정하지 않고
# review_needed.xlsx로 빼서 사람이(교수님과 상의해) 직접 결정하도록 한다. 어깨/팔꿈치
# 표본 자체가 적어서(전체 4,674건 중 900건 안팎) 한 건 한 건이 라벨 분포에 영향이 크다.
AMBIGUOUS_KEYWORDS = [r"biceps", r"triceps", r"deltoid", r"median nerve"]
AMBIGUOUS_PATTERN = re.compile("|".join(AMBIGUOUS_KEYWORDS), re.IGNORECASE)

SHOULDER_PATTERN = re.compile("|".join(SHOULDER_STRICT_KEYWORDS), re.IGNORECASE)
ELBOW_STRICT_PATTERN = re.compile("|".join(ELBOW_STRICT_KEYWORDS), re.IGNORECASE)
ELBOW_BROAD_PATTERN = re.compile(
    "|".join(ELBOW_STRICT_KEYWORDS + ELBOW_BROAD_EXTRA_KEYWORDS), re.IGNORECASE
)


def load_all_transactions() -> pd.DataFrame:
    frames = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = RAW_DIR / f"transactions_{year}.parquet"
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["player_id", "description_raw"])
    return df


def extract_initial_il_candidates(df: pd.DataFrame) -> pd.DataFrame:
    desc = df["description_raw"].fillna("")
    il_mention = desc.str.contains(IL_MENTION_PATTERN)
    placed = desc.str.contains(PLACED_PATTERN)
    excluded = desc.str.contains(EXCLUDE_PATTERN)

    df = df.copy()
    df["il_mention_flag"] = il_mention
    df["initial_il_placement_flag"] = il_mention & placed & ~excluded
    return df[df["initial_il_placement_flag"]].copy()


def filter_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    pos = df["description_raw"].str.extract(POSITION_EXTRACT_PATTERN)[0].str.upper()
    df = df.copy()
    df["position_code"] = pos
    return df[df["position_code"].isin(PITCHER_POSITIONS)].copy()


def dedup_to_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """선수 ID + IL 시작일(effective_date) 기준으로 동일 사건 통합."""
    df = df.copy()
    df["il_start_date"] = df["effective_date"]
    df["desc_len"] = df["description_raw"].str.len()

    group_keys = ["player_id", "il_start_date"]

    agg_info = (
        df.groupby(group_keys)
        .agg(
            all_descriptions=("description_raw", lambda s: " | ".join(s)),
            n_merged_records=("description_raw", "size"),
        )
        .reset_index()
    )

    representative = (
        df.sort_values("desc_len", ascending=False)
        .drop_duplicates(subset=group_keys, keep="first")
        .drop(columns=["desc_len"])
    )

    episodes = representative.merge(agg_info, on=group_keys, how="left")
    episodes = episodes.rename(columns={"date": "date_reported"})
    keep_cols = [
        "player_id", "player_name", "position_code", "il_start_date",
        "date_reported", "description_raw", "all_descriptions", "n_merged_records",
    ]
    return episodes[keep_cols].reset_index(drop=True)


def match_return_dates(episodes: pd.DataFrame, all_txn: pd.DataFrame) -> pd.DataFrame:
    """IL 시작일 이후 동일 선수의 reinstated/activated/returned 기록에서 종료일을 찾는다."""
    return_desc = all_txn["description_raw"].fillna("")
    return_mask = return_desc.str.contains(
        r"reinstated|activated|returned", case=False
    )
    returns = all_txn[return_mask][["player_id", "effective_date"]].dropna()
    returns = returns.sort_values("effective_date")

    episodes = episodes.sort_values("il_start_date").copy()
    end_dates = []
    for _, row in episodes.iterrows():
        cand = returns[
            (returns["player_id"] == row["player_id"])
            & (returns["effective_date"] > row["il_start_date"])
        ]
        end_dates.append(cand["effective_date"].min() if not cand.empty else pd.NaT)
    episodes["il_end_date"] = end_dates
    return episodes


def classify_injuries(episodes: pd.DataFrame) -> pd.DataFrame:
    desc = episodes["description_raw"].fillna("")

    shoulder_hit = desc.str.contains(SHOULDER_PATTERN)
    labrum_hit = desc.str.contains(LABRUM_PATTERN) & ~desc.str.contains(HIP_PATTERN)
    shoulder_strict = shoulder_hit | labrum_hit

    elbow_strict = desc.str.contains(ELBOW_STRICT_PATTERN)
    elbow_broad = desc.str.contains(ELBOW_BROAD_PATTERN)

    # 이미 shoulder/elbow strict 키워드로 명확히 잡힌 건 굳이 다시 review 대상으로
    # 안 뺀다. shoulder/elbow strict 둘 다 아닌데 biceps/triceps/deltoid/median nerve가
    # 나오는 경우만 "애매해서 사람이 봐야 하는 케이스"로 취급한다.
    ambiguous_flag = (
        desc.str.contains(AMBIGUOUS_PATTERN) & ~shoulder_hit & ~labrum_hit & ~elbow_strict
    )

    episodes = episodes.copy()
    episodes["shoulder_strict_flag"] = shoulder_strict
    episodes["elbow_strict_flag"] = elbow_strict
    episodes["elbow_broad_flag"] = elbow_broad
    episodes["both_strict_flag"] = shoulder_strict & elbow_strict
    episodes["both_broad_flag"] = shoulder_strict & elbow_broad
    episodes["ambiguous_muscle_flag"] = ambiguous_flag

    def classify(row, elbow_col, both_col):
        if row[both_col]:
            return 3
        if row["shoulder_strict_flag"]:
            return 1
        if row[elbow_col]:
            return 2
        return 3

    episodes["injury_class_strict"] = episodes.apply(
        lambda r: classify(r, "elbow_strict_flag", "both_strict_flag"), axis=1
    )
    episodes["injury_class_broad"] = episodes.apply(
        lambda r: classify(r, "elbow_broad_flag", "both_broad_flag"), axis=1
    )
    episodes["classification_changed_by_broad"] = (
        episodes["injury_class_strict"] != episodes["injury_class_broad"]
    )

    episodes["surgery_flag"] = desc.str.contains(r"surgery", case=False)
    episodes["recovering_flag"] = desc.str.contains(r"recover", case=False)
    episodes["classification_review_flag"] = (
        episodes["both_strict_flag"] | episodes["both_broad_flag"] | episodes["ambiguous_muscle_flag"]
    )
    return episodes


def main():
    print("[load] transactions 2016~2025 ...")
    all_txn = load_all_transactions()
    print(f"  total transactions: {len(all_txn):,}")

    candidates_all_positions = extract_initial_il_candidates(all_txn)
    print(f"[step] 전체 포지션 최초 IL 후보: {len(candidates_all_positions):,}")

    pitcher_candidates = filter_pitchers(candidates_all_positions)
    print(f"[step] 투수 최초 IL 후보: {len(pitcher_candidates):,}")
    print(pitcher_candidates["position_code"].value_counts())

    episodes = dedup_to_episodes(pitcher_candidates)
    print(f"[step] 중복 제거 후 injury episode: {len(episodes):,}")
    print(f"  고유 투수 수: {episodes['player_id'].nunique():,}")

    before_clip = len(episodes)
    episodes = episodes[
        episodes["il_start_date"].dt.year.between(START_YEAR, END_YEAR)
    ].reset_index(drop=True)
    print(f"  분석기간({START_YEAR}~{END_YEAR}) 외 소급/이월 건 제외: {before_clip} -> {len(episodes)}")

    episodes = match_return_dates(episodes, all_txn)
    match_rate = episodes["il_end_date"].notna().mean()
    print(f"  IL 종료일 매칭률: {match_rate:.2%}")

    episodes = classify_injuries(episodes)

    label_map = {1: "어깨", 2: "팔꿈치", 3: "그 외"}
    for col in ("injury_class_strict", "injury_class_broad"):
        counts = episodes[col].map(label_map).value_counts()
        print(f"\n[{col}]")
        print(counts)

    out_path = OUT_DIR / "injury_episodes.parquet"
    episodes.to_parquet(out_path, index=False)
    print(f"\n[saved] {out_path}")

    review = episodes[episodes["classification_review_flag"]]
    review_path = OUT_DIR / "review_needed.xlsx"
    review.to_excel(review_path, index=False)
    print(f"[saved] {review_path} ({len(review)} rows)")

    yearly = (
        episodes.assign(year=episodes["il_start_date"].dt.year)
        .groupby(["year", "injury_class_strict"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=label_map)
    )
    print("\n[연도별 Strict 기준 결과]")
    print(yearly)


if __name__ == "__main__":
    main()
