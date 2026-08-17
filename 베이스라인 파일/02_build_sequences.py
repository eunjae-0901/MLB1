"""Create the paper-compatible 224 x 102 TJS classification tensor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "필요없는거" / "mlb_injury_project" / "data" / "raw" / "statcast"
DATA = HERE / "data"
PITCH_TYPES = ["CH", "CU", "FC", "FF", "SI", "SL"]
METRICS = ["ax", "ay", "az", "effective_speed", "pfx_x", "pfx_z", "plate_x", "plate_z", "release_extension", "release_pos_x", "release_pos_z", "release_speed", "release_spin_rate", "spin_axis", "vx0", "vy0", "vz0"]
FEATURES = [f"{m}_{p}" for m in METRICS for p in PITCH_TYPES]
GRID = np.arange(100, 1220, 5)  # 100, ..., 1215: 224 bins, as in released code


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--delta-mode", choices=["official_full_period", "paper_first_season"], default="official_full_period")
    return p.parse_args()


def load_pitches() -> pd.DataFrame:
    cols = ["player_name", "game_type", "pitch_type", "game_date", "pitcher", "game_year", "p_throws", *METRICS]
    d = pd.concat([pd.read_parquet(RAW / f"statcast_{y}.parquet", columns=cols) for y in range(2016, 2024)], ignore_index=True)
    d["game_date"] = pd.to_datetime(d["game_date"], errors="coerce")
    return d.query("game_type == 'R' and pitch_type in @PITCH_TYPES").copy()


def reflect_left_handed(d: pd.DataFrame) -> pd.DataFrame:
    left = d["p_throws"].eq("L")
    for col in ("ax", "pfx_x", "release_pos_x", "vx0"):
        d.loc[left, col] = -d.loc[left, col]
    d.loc[left, "spin_axis"] = (360 - d.loc[left, "spin_axis"]) % 360
    return d


def game_pitch_type_means(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby(["pitcher", "player_name", "game_date", "game_year", "pitch_type"], as_index=False)[METRICS].mean()
    wide = g.pivot(index=["pitcher", "player_name", "game_date", "game_year"], columns="pitch_type", values=METRICS)
    wide.columns = [f"{metric}_{pitch}" for metric, pitch in wide.columns]
    wide = wide.reset_index().reindex(columns=["pitcher", "player_name", "game_date", "game_year", *FEATURES])
    wide[FEATURES] = wide[FEATURES].apply(pd.to_numeric, errors="coerce")
    return wide


def baseline_delta(d: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "official_full_period":
        baseline = d.groupby("pitcher")[FEATURES].transform("mean")
    else:
        first_year = d.groupby("pitcher")["game_year"].transform("min")
        first = d.loc[d["game_year"].eq(first_year)].groupby("pitcher")[FEATURES].mean()
        baseline = d[["pitcher"]].join(first, on="pitcher")[FEATURES]
    d[FEATURES] = d[FEATURES] - baseline.to_numpy()
    return d


def tensor_for_pitcher(d: pd.DataFrame, reference: pd.Timestamp, global_mean: pd.Series) -> np.ndarray:
    x = d.copy()
    x["days"] = (reference - x["game_date"]).dt.days
    x = x.loc[x["days"].between(0, 1310)].copy()
    x["bin"] = (x["days"] // 5) * 5
    x = x.groupby("bin")[FEATURES].mean().reindex(np.arange(0, 1311, 5))
    for col in FEATURES:
        x[col] = x[col].interpolate(limit_direction="both")
        x[col] = x[col].fillna(global_mean[col])
    return x.loc[GRID, FEATURES].to_numpy(dtype="float32")


def main() -> None:
    a = args(); cohort = pd.read_csv(DATA / "tjs_cohort.csv")
    cohort["reference_date"] = pd.to_datetime(cohort["reference_date"])
    pitches = load_pitches(); pitches = pitches.loc[pitches["pitcher"].isin(cohort["pitcher"])]
    wide = game_pitch_type_means(reflect_left_handed(pitches))
    refs = cohort.set_index("pitcher")["reference_date"]
    wide = wide.loc[wide["game_date"] <= wide["pitcher"].map(refs)].copy()
    wide = baseline_delta(wide, a.delta_mode)
    med = wide[FEATURES].median(); std = wide[FEATURES].std()
    wide[FEATURES] = wide[FEATURES].mask((wide[FEATURES] < med - 4.7 * std) | (wide[FEATURES] > med + 4.7 * std))
    global_mean = wide[FEATURES].mean()
    arrays, labels, ids, names = [], [], [], []
    for row in cohort.itertuples(index=False):
        pg = wide.loc[wide["pitcher"] == row.pitcher]
        if pg.empty:
            continue
        arrays.append(tensor_for_pitcher(pg, row.reference_date, global_mean))
        labels.append(row.target); ids.append(row.pitcher)
        names.append(pg["player_name"].dropna().iloc[0] if pg["player_name"].notna().any() else str(row.pitcher))
    X = np.stack(arrays); y = np.asarray(labels, dtype="int8")
    np.savez_compressed(DATA / f"tjs_sequences_{a.delta_mode}.npz", X=X, y=y, pitcher_id=np.asarray(ids), player_name=np.asarray(names), feature_names=np.asarray(FEATURES), day_bins=GRID)
    manifest = {"delta_mode": a.delta_mode, "shape": list(X.shape), "positive": int(y.sum()), "negative": int((y == 0).sum()), "day_bins": [int(GRID.min()), int(GRID.max()), 5], "paper_expected_shape": [620, 224, 102]}
    (DATA / f"sequence_manifest_{a.delta_mode}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
