"""Domain-driven feature engineering for the 05 methodology.

Builds acute:chronic workload ratios, rest-day patterns, velocity/spin/
extension trend deltas, and season/career cumulative workload on top of
the same leakage-safe game history used by 02_build_100d_sequences.py
(only games with game_date <= snapshot date are ever used).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

WINDOWS = (7, 14, 28, 60, 100)


def player_block(g: pd.DataFrame, snap_dates: np.ndarray) -> pd.DataFrame:
    dates = g["game_date"].to_numpy("datetime64[D]")
    pitches = g["n_pitches"].to_numpy("float64")
    outs = g["outs_recorded"].to_numpy("float64")
    starts = g["is_start"].astype("float64").to_numpy()
    velo_sum = g["sum_v_all"].to_numpy("float64")
    ext_sum = g["sum_ext_all"].to_numpy("float64")
    spin_sum = g["sum_spin_all"].to_numpy("float64")
    fb_n = g["n_FB"].to_numpy("float64")
    rest = g["days_since_prev_game"].to_numpy("float64")
    rest_valid = np.nan_to_num(rest, nan=0.0)
    rest_mask = (~np.isnan(rest)).astype("float64")
    short_rest = ((rest < 4) & ~np.isnan(rest)).astype("float64")
    years = g["game_date"].dt.year.to_numpy()

    zero = np.zeros(1)
    cum_pitches = np.concatenate([zero, np.cumsum(pitches)])
    cum_outs = np.concatenate([zero, np.cumsum(outs)])
    cum_starts = np.concatenate([zero, np.cumsum(starts)])
    cum_velo = np.concatenate([zero, np.cumsum(velo_sum)])
    cum_ext = np.concatenate([zero, np.cumsum(ext_sum)])
    cum_spin = np.concatenate([zero, np.cumsum(spin_sum)])
    cum_fb = np.concatenate([zero, np.cumsum(fb_n)])
    cum_rest = np.concatenate([zero, np.cumsum(rest_valid)])
    cum_rest_sq = np.concatenate([zero, np.cumsum(rest_valid ** 2)])
    cum_rest_n = np.concatenate([zero, np.cumsum(rest_mask)])
    cum_short_rest = np.concatenate([zero, np.cumsum(short_rest)])

    rows = []
    for sd in snap_dates:
        hi = int(np.searchsorted(dates, sd, side="right"))
        row: dict[str, float] = {}
        win_pitches, win_outs, win_velo = {}, {}, {}
        for w in WINDOWS:
            lo = int(np.searchsorted(dates, sd - np.timedelta64(w - 1, "D"), side="left"))
            p = cum_pitches[hi] - cum_pitches[lo]
            o = cum_outs[hi] - cum_outs[lo]
            apps = hi - lo
            st = cum_starts[hi] - cum_starts[lo]
            v = (cum_velo[hi] - cum_velo[lo]) / p if p > 0 else np.nan
            e = (cum_ext[hi] - cum_ext[lo]) / p if p > 0 else np.nan
            sp = (cum_spin[hi] - cum_spin[lo]) / p if p > 0 else np.nan
            fb = (cum_fb[hi] - cum_fb[lo]) / p if p > 0 else np.nan
            row[f"pitches_{w}d"] = p
            row[f"outs_{w}d"] = o
            row[f"apps_{w}d"] = float(apps)
            row[f"start_frac_{w}d"] = st / apps if apps > 0 else np.nan
            row[f"velo_{w}d"] = v
            row[f"ext_{w}d"] = e
            row[f"spin_{w}d"] = sp
            row[f"fb_share_{w}d"] = fb
            win_pitches[w], win_outs[w], win_velo[w] = p, o, v
        eps = 1e-6
        row["acwr_pitches_7_28"] = (win_pitches[7] / 7) / max(win_pitches[28] / 28, eps)
        row["acwr_outs_7_28"] = (win_outs[7] / 7) / max(win_outs[28] / 28, eps)
        row["acwr_pitches_14_60"] = (win_pitches[14] / 14) / max(win_pitches[60] / 60, eps)
        row["velo_delta_14_100"] = (win_velo[14] - win_velo[100]) if np.isfinite(win_velo[14]) and np.isfinite(win_velo[100]) else np.nan
        row["ext_delta_14_100"] = row[f"ext_14d"] - row[f"ext_100d"]
        row["spin_delta_14_100"] = row[f"spin_14d"] - row[f"spin_100d"]
        row["fb_share_delta_14_100"] = row[f"fb_share_14d"] - row[f"fb_share_100d"]

        lo100 = int(np.searchsorted(dates, sd - np.timedelta64(99, "D"), side="left"))
        n100 = cum_rest_n[hi] - cum_rest_n[lo100]
        row["rest_mean_100d"] = (cum_rest[hi] - cum_rest[lo100]) / n100 if n100 > 0 else np.nan
        mean100 = row["rest_mean_100d"]
        var100 = (cum_rest_sq[hi] - cum_rest_sq[lo100]) / n100 - mean100 ** 2 if n100 > 0 and not np.isnan(mean100) else np.nan
        row["rest_std_100d"] = np.sqrt(max(var100, 0)) if not np.isnan(var100) else np.nan
        lo30 = int(np.searchsorted(dates, sd - np.timedelta64(29, "D"), side="left"))
        row["short_rest_count_30d"] = cum_short_rest[hi] - cum_short_rest[lo30]

        row["career_pitches"] = cum_pitches[hi]
        row["career_outs"] = cum_outs[hi]
        row["career_apps"] = float(hi)
        row["career_start_frac"] = cum_starts[hi] / hi if hi > 0 else np.nan

        year = sd.astype("datetime64[Y]").astype(int) + 1970
        season_start = np.datetime64(f"{year}-01-01")
        lo_season = int(np.searchsorted(dates, season_start, side="left"))
        row["season_pitches"] = cum_pitches[hi] - cum_pitches[lo_season]
        row["season_outs"] = cum_outs[hi] - cum_outs[lo_season]
        row["season_apps"] = float(hi - lo_season)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    games = pd.read_parquet(DATA / "pitcher_game_features.parquet").sort_values(["player_id", "game_date"])
    snaps = pd.read_parquet(DATA / "dynamic_snapshot_targets.parquet").sort_values(["player_id", "game_date"]).reset_index(drop=True)

    game_groups = {int(k): v for k, v in games.groupby("player_id", sort=False)}
    blocks = []
    for pid, idx in snaps.groupby("player_id", sort=False).groups.items():
        g = game_groups.get(int(pid))
        sub = snaps.loc[idx]
        if g is None or len(g) == 0:
            blocks.append(pd.DataFrame(index=sub.index))
            continue
        snap_dates = sub["game_date"].to_numpy("datetime64[D]")
        feats = player_block(g, snap_dates)
        feats.index = idx
        blocks.append(feats)
    engineered = pd.concat(blocks).reindex(snaps.index)

    meta_cols = ["player_id", "game_date", "split", "role", "target_100d", "regression_days",
                 "event_weight", "evaluation_cohort", "seen_in_train", "past_arm_il_count",
                 "days_since_last_arm_il", "age", "p_throws"]
    out = pd.concat([snaps[meta_cols].reset_index(drop=True), engineered.reset_index(drop=True)], axis=1)
    out.to_parquet(DATA / "engineered_features_05.parquet", index=False)
    print("shape:", out.shape)
    print("feature columns:", [c for c in out.columns if c not in meta_cols])
    print(out.isna().mean().sort_values(ascending=False).head(15))


if __name__ == "__main__":
    main()
