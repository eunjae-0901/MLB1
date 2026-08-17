"""Build leakage-safe 20 x 5-day sequences for all dynamic snapshots."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
N_BINS, BIN_DAYS = 20, 5


def safe_div(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype="float64"), where=b > 0)


def daily_features(games: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = games[["player_id", "game_date", "n_pitches", "n_batters_faced", "outs_recorded",
                 "days_since_prev_game", "age", "is_start"]].copy()
    out["appearances"] = 1.0
    out["is_start"] = out["is_start"].astype(float)
    for name, numerator, denominator in [
        ("velo", "sum_v_all", "n_pitches"), ("plate_x", "sum_x_all", "n_pitches"),
        ("plate_z", "sum_z_all", "n_pitches"), ("extension", "sum_ext_all", "n_pitches"),
        ("spin", "sum_spin_all", "n_pitches"),
    ]:
        out[name] = safe_div(games[numerator].to_numpy(float), games[denominator].to_numpy(float))
    for pitch in ["FB", "SI", "CT", "SL", "CB", "CH", "SP"]:
        count = games[f"n_{pitch}"].to_numpy(float)
        out[f"usage_{pitch}"] = count / np.maximum(games["n_pitches"].to_numpy(float), 1)
        out[f"velo_{pitch}"] = safe_div(games[f"sum_v{pitch}"].to_numpy(float), count)
        out[f"spin_{pitch}"] = safe_div(games[f"sum_spin{pitch}"].to_numpy(float), count)
    feature_cols = [c for c in out if c not in {"player_id", "game_date"}]
    return out, feature_cols


def build_sequences(games: pd.DataFrame, snapshots: pd.DataFrame):
    daily, feature_cols = daily_features(games)
    daily = daily.sort_values(["player_id", "game_date"])
    snapshots = snapshots.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    sequences = np.full((len(snapshots), N_BINS, len(feature_cols)), np.nan, dtype="float32")
    masks = np.zeros((len(snapshots), N_BINS), dtype=bool)
    game_groups = {int(k): v for k, v in daily.groupby("player_id", sort=False)}
    for pid, indices in snapshots.groupby("player_id", sort=False).groups.items():
        history = game_groups.get(int(pid))
        if history is None:
            continue
        dates = history["game_date"].to_numpy(dtype="datetime64[D]")
        values = history[feature_cols].to_numpy(dtype="float64")
        for row_index in indices:
            cutoff = np.datetime64(snapshots.at[row_index, "game_date"].date())
            age_days = (cutoff - dates).astype(int)
            valid = (age_days >= 0) & (age_days < N_BINS * BIN_DAYS)
            for b in range(N_BINS):
                take = valid & (age_days >= b * BIN_DAYS) & (age_days < (b + 1) * BIN_DAYS)
                if take.any():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        sequences[row_index, b] = np.nanmean(values[take], axis=0)
                    masks[row_index, b] = True
    return snapshots, sequences, masks, feature_cols


def main():
    games = pd.read_parquet(DATA / "pitcher_game_features.parquet")
    snapshots = pd.read_parquet(DATA / "dynamic_snapshot_targets.parquet")
    snapshots, x, mask, features = build_sequences(games, snapshots)
    meta_cols = ["player_id", "game_date", "split", "role", "target_100d", "regression_days",
                 "event_weight", "evaluation_cohort", "seen_in_train", "past_arm_il_count",
                 "days_since_last_arm_il", "age", "p_throws"]
    meta = snapshots[meta_cols].copy()
    meta.to_parquet(DATA / "sequence_metadata.parquet", index=False)
    np.savez_compressed(DATA / "sequences_100d_5d.npz", X=x, mask=mask,
                        feature_names=np.asarray(features, dtype=str))
    manifest = {"shape": list(x.shape), "n_masked_bins": int(mask.sum()), "features": features,
                "bin_order": "index 0=current 0-4 days; index 19=95-99 days ago",
                "input_cutoff": "snapshot date inclusive"}
    (RESULTS / "sequence_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
