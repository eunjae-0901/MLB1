"""
LSTM/Transformer 같은 시계열 딥러닝 모델에 바로 넣을 수 있는 numpy 배열(.npz)을 만든다.
(GPU 딥러닝 학습 자체는 별도 가상환경에서 models/03~05_*.py로 실행할 것 - 이 스크립트는
데이터 준비만 하며 duckdb/pandas/numpy만 있으면 되고 torch는 필요 없다.)

각 관측치(투수 1명의 특정 시점)마다:
  - X_seq: 최근 K경기의 경기별 지표를 시간순(오래된 것 -> 최신 것)으로 쌓은 배열
           (선발 K=3, 불펜 K=5). 부족한 경기는 앞쪽을 0으로 패딩.
  - mask : 그 timestep이 진짜 데이터인지(1) 패딩인지(0)
  - X_static: 나이/키/몸무게/휴식일 등 시점 하나에 대한 정적 변수
  - cat_static: 범주형(투구손, 출신국) 정수 인코딩 (임베딩용)
  - y, split: 기존 rolling-window 데이터셋에서 그대로 재사용

출력: data/processed/{role}_sequence_arrays.npz + {role}_categories.json (임베딩 vocab 크기 확인용)
"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_defs import METRICS, PITCH_GROUPS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PITCHER_GAME_PATH = str(
    PROJECT_ROOT / "data" / "interim" / "pitcher_game" / "pitcher_game_role.parquet"
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

N_LAGS = {"starter": 3, "bullpen": 5}

PER_GAME_FEATURES = [f"g_{name}_all" for name, _, _ in METRICS] + ["g_n_pitches"]
for _g in PITCH_GROUPS:
    PER_GAME_FEATURES.append(f"g_pct_{_g}")
    for _name, _, _ in METRICS:
        PER_GAME_FEATURES.append(f"g_{_name}{_g}")


def per_game_feature_sql() -> str:
    cols = [f"sum_{name}_all / NULLIF(n{name}_all, 0) AS g_{name}_all" for name, _, _ in METRICS]
    cols.append("n_pitches AS g_n_pitches")
    for g in PITCH_GROUPS:
        cols.append(f"n_{g} / NULLIF(n_pitches, 0) AS g_pct_{g}")
        for name, _, _ in METRICS:
            cols.append(f"sum_{name}{g} / NULLIF(n{name}{g}, 0) AS g_{name}{g}")
    return ",\n        ".join(cols)


def load_per_game(con: duckdb.DuckDBPyConnection, role: str) -> pd.DataFrame:
    is_start_filter = "true" if role == "starter" else "false"
    query = f"""
        SELECT player_id, game_date,
            {per_game_feature_sql()}
        FROM read_parquet('{PITCHER_GAME_PATH}')
        WHERE is_start = {is_start_filter}
        ORDER BY player_id, game_date
    """
    return con.execute(query).df()


def build_sequences(per_game: pd.DataFrame, anchors: pd.DataFrame, k: int):
    """anchors: player_id, window_end_date(=그 시점의 game_date)가 있는 행들.
    각 anchor에 대해 그 시점까지의 최근 k경기를 모아 (N,k,F) 배열을 만든다."""
    feat_cols = PER_GAME_FEATURES
    per_game_by_player = {pid: g.reset_index(drop=True)
                           for pid, g in per_game.groupby("player_id")}

    n = len(anchors)
    X_seq = np.zeros((n, k, len(feat_cols)), dtype=np.float32)
    mask = np.zeros((n, k), dtype=np.float32)

    for i, row in enumerate(anchors.itertuples(index=False)):
        pid, anchor_date = row.player_id, row.window_end_date
        g = per_game_by_player.get(pid)
        if g is None:
            continue
        sub = g[g["game_date"] <= anchor_date].tail(k)
        vals = sub[feat_cols].to_numpy(dtype=np.float32)
        n_have = len(sub)
        if n_have > 0:
            X_seq[i, k - n_have:, :] = np.nan_to_num(vals, nan=0.0)
            mask[i, k - n_have:] = 1.0

    return X_seq, mask, feat_cols


def build_role(con: duckdb.DuckDBPyConnection, role: str):
    print(f"\n[{role}] 준비 시작")
    k = N_LAGS[role]

    window_df = pd.read_parquet(PROCESSED_DIR / f"{role}_window_dataset.parquet")
    per_game = load_per_game(con, role)
    per_game["game_date"] = pd.to_datetime(per_game["game_date"])
    window_df["window_end_date"] = pd.to_datetime(window_df["window_end_date"])

    anchors = window_df[["player_id", "window_end_date"]]
    X_seq, mask, feat_cols = build_sequences(per_game, anchors, k)
    print(f"  X_seq shape={X_seq.shape}, mask shape={mask.shape}")

    # 범주형 정수 인코딩 (임베딩용 vocab). 0은 '알수없음/미확인'으로 예약.
    p_throws_vocab = {v: idx + 1 for idx, v in enumerate(sorted(window_df["p_throws"].dropna().unique()))}
    country_vocab = {v: idx + 1 for idx, v in enumerate(sorted(window_df["birth_country"].dropna().unique()))}

    cat_p_throws = window_df["p_throws"].map(p_throws_vocab).fillna(0).astype(np.int64).to_numpy()
    cat_country = window_df["birth_country"].map(country_vocab).fillna(0).astype(np.int64).to_numpy()

    static_cols = ["age", "height_inches", "weight_lb", "days_since_prev_game"]
    X_static = window_df[static_cols].astype(np.float32).fillna(0.0).to_numpy()

    y = window_df["label"].astype(np.int64).to_numpy()
    split = window_df["split"].astype(str).to_numpy()

    out_path = PROCESSED_DIR / f"{role}_sequence_arrays.npz"
    np.savez_compressed(
        out_path,
        X_seq=X_seq, mask=mask, X_static=X_static,
        cat_p_throws=cat_p_throws, cat_country=cat_country,
        y=y, split=split,
    )
    print(f"  [saved] {out_path}")

    meta = {
        "role": role,
        "k": k,
        "seq_feature_names": feat_cols,
        "static_feature_names": static_cols,
        "p_throws_vocab_size": len(p_throws_vocab) + 1,
        "country_vocab_size": len(country_vocab) + 1,
        "p_throws_vocab": p_throws_vocab,
        "country_vocab": country_vocab,
        "n_classes": 4,
        "class_counts": {int(c): int((y == c).sum()) for c in sorted(set(y.tolist()))},
        "split_counts": {s: int((split == s).sum()) for s in sorted(set(split.tolist()))},
    }
    meta_path = PROCESSED_DIR / f"{role}_sequence_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [saved] {meta_path}")


def main():
    con = duckdb.connect()
    for role in ("bullpen", "starter"):
        build_role(con, role)
    con.close()


if __name__ == "__main__":
    main()
