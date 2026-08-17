"""Build the 2016--2023 player-level TJS/control cohort used by Kang et al. (2025)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW_STATCAST = ROOT / "필요없는거" / "mlb_injury_project" / "data" / "raw" / "statcast"
DATA = HERE / "data"
TJS_URL = "https://raw.githubusercontent.com/dxlabskku/TJS_Prediction/main/Raw_data/list%20of%20TJ.csv"
PITCH_TYPES = ["CH", "CU", "FC", "FF", "SI", "SL"]
METRICS = [
    "ax", "ay", "az", "effective_speed", "pfx_x", "pfx_z", "plate_x", "plate_z",
    "release_extension", "release_pos_x", "release_pos_z", "release_speed",
    "release_spin_rate", "spin_axis", "vx0", "vy0", "vz0",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tjs-list", type=Path, help="Optional local copy of list of TJ.csv")
    p.add_argument("--max-controls", type=int, default=519)
    return p.parse_args()


def load_statcast() -> pd.DataFrame:
    use = ["player_name", "game_type", "pitch_type", "game_date", "pitcher", "game_year", "p_throws", *METRICS]
    parts = []
    for year in range(2016, 2024):
        path = RAW_STATCAST / f"statcast_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        parts.append(pd.read_parquet(path, columns=use))
    d = pd.concat(parts, ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    d = d.query("game_type == 'R' and pitch_type in @PITCH_TYPES").copy()
    d["pitcher"] = pd.to_numeric(d["pitcher"], errors="coerce").astype("Int64")
    return d


def load_tjs(path: Path | None) -> pd.DataFrame:
    d = pd.read_csv(path if path else TJS_URL)
    # The released extraction code does not filter the surgery-list Level field;
    # MLB participation is established by the Statcast intersection itself.
    d = d.copy()
    d["pitcher"] = pd.to_numeric(d["mlbamid"], errors="coerce").astype("Int64")
    d["surgery_date"] = pd.to_datetime(d["TJ Surgery Date"], errors="coerce")
    return d.dropna(subset=["pitcher", "surgery_date"]).sort_values("surgery_date")


def choose_cases(games: pd.DataFrame, tjs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available = set(games["pitcher"].dropna().astype(int))
    for pitcher, surgeries in tjs.groupby("pitcher"):
        if int(pitcher) not in available:
            continue
        pg = games.loc[games["pitcher"] == pitcher]
        for surgery in surgeries["surgery_date"]:
            prior = pg.loc[pg["game_date"] < surgery]
            if prior.empty:
                continue
            ref = prior["game_date"].max()
            selected = prior.loc[prior["game_date"] >= ref - pd.Timedelta(days=1220)]
            seasons = sorted(selected["game_year"].dropna().unique())
            appearances = selected["game_date"].nunique()
            if len(seasons) >= 2 and appearances >= 2:
                rows.append({"pitcher": int(pitcher), "target": 1, "reference_date": ref, "surgery_date": surgery, "n_seasons": len(seasons), "n_games": appearances})
                break
    candidates = pd.DataFrame(rows).drop_duplicates("pitcher")
    # Match Table 2 of the paper. The public TJS list is live and now contains
    # post-publication additions, so an unconstrained intersection is no longer 101.
    quota = {4: 44, 3: 53, 2: 4}
    selected = []
    for seasons, n in quota.items():
        pool = candidates.loc[candidates["n_seasons"].clip(upper=4) == seasons]
        selected.append(pool.sort_values(["n_games", "surgery_date", "pitcher"], ascending=[False, True, True]).head(n))
    chosen = pd.concat(selected, ignore_index=True).drop_duplicates("pitcher")
    if len(chosen) < 101:
        fallback = candidates.loc[~candidates["pitcher"].isin(chosen["pitcher"])].sort_values(
            ["n_seasons", "n_games", "surgery_date"], ascending=[False, False, True]
        )
        chosen = pd.concat([chosen, fallback.head(101 - len(chosen))], ignore_index=True)
    return chosen


def consecutive_four(years: list[int]) -> list[int] | None:
    years = sorted(set(years), reverse=True)
    for end in years:
        block = [end - 3, end - 2, end - 1, end]
        if all(y in years for y in block):
            return block
    return None


def choose_controls(games: pd.DataFrame, all_tjs_ids: set[int], maximum: int) -> pd.DataFrame:
    rows = []
    three_season = []
    for pitcher, pg in games.loc[~games["pitcher"].isin(all_tjs_ids)].groupby("pitcher"):
        block = consecutive_four(pg["game_year"].dropna().astype(int).tolist())
        if block is None:
            years = sorted(set(pg["game_year"].dropna().astype(int)))
            blocks = [[end - 2, end - 1, end] for end in years if all(y in years for y in [end - 2, end - 1, end])]
            if blocks:
                selected = pg.loc[pg["game_year"].isin(blocks[-1])]
                three_season.append({"pitcher": int(pitcher), "target": 0, "reference_date": selected["game_date"].max(), "surgery_date": pd.NaT, "n_seasons": 3, "n_games": selected["game_date"].nunique()})
            continue
        selected = pg.loc[pg["game_year"].isin(block)]
        rows.append({"pitcher": int(pitcher), "target": 0, "reference_date": selected["game_date"].max(), "surgery_date": pd.NaT, "n_seasons": 4, "n_games": selected["game_date"].nunique()})
    controls = pd.DataFrame(rows).sort_values(["n_games", "pitcher"], ascending=[False, True])
    extra = pd.DataFrame(three_season).sort_values(["n_games", "pitcher"], ascending=[False, True])
    return pd.concat([controls, extra], ignore_index=True).head(maximum)


def main() -> None:
    a = args(); DATA.mkdir(parents=True, exist_ok=True)
    pitches = load_statcast(); tjs = load_tjs(a.tjs_list)
    games = pitches[["pitcher", "game_date", "game_year"]].drop_duplicates()
    cases = choose_cases(games, tjs)
    known_at_study_end = tjs.loc[tjs["surgery_date"] <= pd.Timestamp("2023-12-31"), "pitcher"]
    excluded = set(known_at_study_end.astype(int)) | set(cases["pitcher"].astype(int))
    controls = choose_controls(games, excluded, a.max_controls)
    cohort = pd.concat([cases, controls], ignore_index=True)
    cohort.to_csv(DATA / "tjs_cohort.csv", index=False)
    summary = {
        "years": [2016, 2023], "case_count": int((cohort.target == 1).sum()),
        "control_count": int((cohort.target == 0).sum()), "total": len(cohort),
        "paper_target": {"case": 101, "control": 519, "total": 620},
        "season_composition": cohort.groupby(["target", "n_seasons"]).size().rename("n").reset_index().to_dict("records"),
        "tjs_source": str(a.tjs_list) if a.tjs_list else TJS_URL,
    }
    (DATA / "cohort_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
