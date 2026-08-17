"""Collect 5-day dynamic snapshots and match future arm-IL targets without input leakage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "필요없는거" / "mlb_injury_project" / "data"
GAME_PATH = SOURCE / "interim" / "pitcher_game" / "pitcher_game_role.parquet"
EPISODE_PATH = SOURCE / "interim" / "injury_episodes" / "injury_episodes.parquet"
DATA = HERE / "data"
RESULTS = HERE / "results"
HORIZONS = (30, 60, 100)


def split_for_date(date: pd.Timestamp) -> str:
    if date.year <= 2021:
        return "train"
    if date.year <= 2023:
        return "validation"
    if date.year <= 2025:
        return "test"
    return "out_of_scope"


def select_five_day_snapshots(games: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, group in games.sort_values(["player_id", "game_date", "game_pk"]).groupby("player_id"):
        last = None
        for row in group.itertuples(index=False):
            if last is None or (row.game_date - last).days >= 5:
                selected.append(row)
                last = row.game_date
    return pd.DataFrame(selected, columns=games.columns)


def prepare_games() -> pd.DataFrame:
    games = pd.read_parquet(GAME_PATH)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce").astype("datetime64[ns]")
    games["player_id"] = pd.to_numeric(games["player_id"], errors="coerce")
    games = games.dropna(subset=["player_id", "game_date"])
    games["player_id"] = games["player_id"].astype("int64")
    games = games.loc[games["game_date"].dt.year.between(2016, 2025)].copy()
    # One row per pitcher/date. Doubleheaders are combined using additive raw columns;
    # role is starter if any appearance that day was a start.
    sum_cols = [c for c in games.columns if c.startswith(("sum_", "n_")) or c == "outs_recorded"]
    first_cols = [c for c in ["p_throws", "age", "pitcher_team", "days_since_prev_game"] if c in games]
    agg = {**{c: "sum" for c in sum_cols}, **{c: "first" for c in first_cols}, "is_start": "max", "is_complete_game": "max", "game_pk": "first"}
    daily = games.groupby(["player_id", "game_date"], as_index=False).agg(agg)
    daily["role"] = np.where(daily["is_start"], "starter", "bullpen")
    return daily.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def prepare_arm_events() -> pd.DataFrame:
    episodes = pd.read_parquet(EPISODE_PATH)
    episodes["player_id"] = pd.to_numeric(episodes["player_id"], errors="coerce")
    episodes["il_start_date"] = pd.to_datetime(episodes["il_start_date"], errors="coerce").astype("datetime64[ns]")
    events = episodes.loc[episodes["injury_class_strict"].isin([1, 2])].copy()
    events = events.dropna(subset=["player_id", "il_start_date"]).copy()
    events["player_id"] = events["player_id"].astype("int64")
    events = events.sort_values(["player_id", "il_start_date"])
    events["event_id"] = events["player_id"].astype(str) + "_" + events["il_start_date"].dt.strftime("%Y%m%d") + "_" + events["injury_class_strict"].astype(int).astype(str)
    return events[["player_id", "il_start_date", "injury_class_strict", "event_id", "surgery_flag", "recovering_flag"]]


def match_next_event(snapshots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    # merge_asof requires global sorting by the time key first.
    left = snapshots.sort_values(["game_date", "player_id"]).copy()
    right = events.rename(columns={"il_start_date": "next_il_date"}).sort_values(["next_il_date", "player_id"])
    matched = pd.merge_asof(
        left, right, left_on="game_date", right_on="next_il_date", by="player_id",
        direction="forward", allow_exact_matches=False,
    )
    matched["days_to_next_arm_il"] = (matched["next_il_date"] - matched["game_date"]).dt.days
    observation_cutoff = events["il_start_date"].max()
    for horizon in HORIZONS:
        positive = matched["days_to_next_arm_il"].between(1, horizon)
        fully_observed = matched["game_date"].add(pd.Timedelta(days=horizon)).le(observation_cutoff)
        matched[f"observed_{horizon}d"] = fully_observed
        matched[f"target_{horizon}d"] = pd.Series(
            np.where(fully_observed, positive.astype("int8"), pd.NA),
            index=matched.index, dtype="Int8",
        )
    positive_100d = matched["target_100d"].eq(1).fillna(False)
    matched["regression_days"] = matched["days_to_next_arm_il"].where(positive_100d)
    matched["split"] = matched["game_date"].map(split_for_date)
    matched = matched.loc[matched["split"] != "out_of_scope"].copy()
    counts = matched.loc[positive_100d].groupby("event_id").size()
    matched["event_positive_snapshots"] = matched["event_id"].map(counts).where(positive_100d)
    matched["event_weight"] = np.where(positive_100d, 1 / matched["event_positive_snapshots"], 1.0)
    return matched.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def add_past_history(snapshots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = snapshots.copy()
    out["past_arm_il_count"] = 0
    out["days_since_last_arm_il"] = np.nan
    event_groups = {int(pid): g["il_start_date"].sort_values().to_numpy(dtype="datetime64[D]") for pid, g in events.groupby("player_id")}
    for pid, idx in out.groupby("player_id").groups.items():
        dates = event_groups.get(int(pid))
        if dates is None:
            continue
        for i in idx:
            day = np.datetime64(out.at[i, "game_date"].date())
            prior = dates[dates < day]
            out.at[i, "past_arm_il_count"] = len(prior)
            if len(prior):
                out.at[i, "days_since_last_arm_il"] = int((day - prior.max()).astype(int))
    return out


def add_generalization_cohorts(matched: pd.DataFrame) -> pd.DataFrame:
    """Flag whether each validation/test pitcher was observed in Train."""
    out = matched.copy()
    train_pitchers = set(out.loc[out["split"].eq("train"), "player_id"])
    out["seen_in_train"] = out["player_id"].isin(train_pitchers)
    out["evaluation_cohort"] = np.select(
        [out["split"].eq("train"), out["seen_in_train"]],
        ["train", "seen_player"],
        default="new_player",
    )
    return out


def summaries(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (split, role), group in matched.groupby(["split", "role"]):
        for horizon in HORIZONS:
            y = group[f"target_{horizon}d"].dropna().astype("int8")
            n_positive = int(y.sum())
            n_negative = int(len(y) - n_positive)
            rows.append({"split": split, "role": role, "horizon_days": horizon, "n_snapshots_total": len(group), "n_labeled": len(y), "n_censored": int(group[f"target_{horizon}d"].isna().sum()), "n_positive": n_positive, "n_negative": n_negative, "positive_rate": float(y.mean()), "negative_to_positive_ratio": float(n_negative/n_positive) if n_positive else np.inf, "n_pitchers": group.loc[y.index, "player_id"].nunique(), "n_positive_events": group.loc[y.index[y.eq(1)], "event_id"].nunique()})
    event = matched.loc[matched["target_100d"].eq(1).fillna(False)].groupby(["split", "event_id", "player_id", "next_il_date", "injury_class_strict"], as_index=False).agg(n_positive_snapshots=("game_date", "size"), earliest_lead_days=("days_to_next_arm_il", "max"), latest_lead_days=("days_to_next_arm_il", "min"), total_event_weight=("event_weight", "sum"))
    return pd.DataFrame(rows), event


def cohort_summary(matched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    evaluation = matched.loc[matched["split"].isin(["validation", "test"])]
    for (split, cohort, role), group in evaluation.groupby(["split", "evaluation_cohort", "role"]):
        y = group["target_100d"].dropna().astype("int8")
        positive = int(y.sum())
        negative = int(len(y) - positive)
        rows.append({
            "split": split, "evaluation_cohort": cohort, "role": role,
            "n_snapshots_total": len(group), "n_labeled_100d": len(y),
            "n_censored_100d": int(group["target_100d"].isna().sum()),
            "n_pitchers": int(group.loc[y.index, "player_id"].nunique()),
            "n_positive": positive, "n_negative": negative,
            "positive_rate": float(y.mean()) if len(y) else np.nan,
            "negative_to_positive_ratio": float(negative / positive) if positive else np.inf,
        })
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    games = prepare_games(); events = prepare_arm_events()
    snapshots = select_five_day_snapshots(games)
    matched = add_generalization_cohorts(add_past_history(match_next_event(snapshots, events), events))
    balance, coverage = summaries(matched)
    cohorts = cohort_summary(matched)
    games.to_parquet(DATA / "pitcher_game_features.parquet", index=False)
    matched.to_parquet(DATA / "dynamic_snapshot_targets.parquet", index=False)
    balance.to_csv(RESULTS / "target_balance.csv", index=False)
    coverage.to_csv(RESULTS / "event_coverage.csv", index=False)
    cohorts.to_csv(RESULTS / "generalization_cohorts.csv", index=False)
    audit = {
        "n_game_rows": len(games), "n_snapshots": len(matched), "n_pitchers": int(matched.player_id.nunique()),
        "n_strict_arm_events_source": len(events), "duplicate_player_date": int(matched.duplicated(["player_id", "game_date"]).sum()),
        "future_target_min_days": int(matched.loc[matched.target_100d.eq(1), "days_to_next_arm_il"].min()),
        "future_target_max_days": int(matched.loc[matched.target_100d.eq(1), "days_to_next_arm_il"].max()),
        "injury_observation_cutoff": events["il_start_date"].max().date().isoformat(),
        "censored_100d_snapshots": int(matched["target_100d"].isna().sum()),
        "input_cutoff_rule": "game_date and prior only", "split_rule": "train=2016-2021; validation=2022-2023; test=2024-2025",
    }
    (RESULTS / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nTarget balance")
    print(balance.to_string(index=False))
    print("\nSeen/new-player evaluation cohorts (100d)")
    print(cohorts.to_string(index=False))
    print("\nAudit")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
