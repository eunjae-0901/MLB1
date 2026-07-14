"""
'평균으로 뭉개지 않고 경기별로 순서를 살린' 버전의 데이터셋을 만든다.
DNN(models/03~05)은 이 parquet을 안 쓰고 05_build_sequence_arrays.py가 만드는 .npz를 쓴다 -
이 스크립트의 결과물(*_sequence_dataset.parquet)은 그 .npz 안에 뭐가 들었는지 사람이 엑셀로
확인하기 위한 참고용이다 (06_export_for_review.py가 엑셀로 export할 때 이 parquet을 읽는다).

기존 rolling-window 데이터셋(bullpen/starter_window_dataset.parquet)은 최근 N일/N경기를
평균 내서 한 줄로 만들었는데, 여기서는 평균 대신 "가장 최근 경기 지표 / 그 전 경기 지표 /
그 전전 경기 지표..."를 각각 별도 컬럼으로 펼쳐서(LAG) 순서 정보를 보존한다.

라벨(label)/분할(split)/인적사항(age, height 등)은 이미 만들어둔
bullpen/starter_window_dataset.parquet에서 그대로 가져와 재사용한다(같은 시점 기준이므로).
"""
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feature_defs import METRICS, PITCH_GROUPS  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PITCHER_GAME_PATH = str(
    PROJECT_ROOT / "data" / "interim" / "pitcher_game" / "pitcher_game_role.parquet"
)
WINDOW_DATASET_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# 선발: 최근 3경기(=기존 window와 동일 개수), 불펜: 최근 5회 등판(14일 window에 들어가는
# 평균적인 등판 수와 비슷한 수준으로 고정 - 날짜가 아니라 "횟수" 기준으로 바꾼 것)
N_LAGS = {"starter": 3, "bullpen": 5}


def per_game_feature_cols() -> str:
    """경기 1개 단위의 지표(평균/비율) 컬럼 생성 SQL. window 롤링 없이 그 경기 자체 값만."""
    cols = []
    for name, _, _ in METRICS:
        cols.append(f"sum_{name}_all / NULLIF(n{name}_all, 0) AS g_{name}_all")
    for g in PITCH_GROUPS:
        cols.append(f"n_{g} / NULLIF(n_pitches, 0) AS g_pct_{g}")
        for name, _, _ in METRICS:
            cols.append(f"sum_{name}{g} / NULLIF(n{name}{g}, 0) AS g_{name}{g}")
    return ",\n        ".join(cols)


def build_role(con: duckdb.DuckDBPyConnection, role: str):
    n_lags = N_LAGS[role]
    is_start_filter = "true" if role == "starter" else "false"

    con.execute(f"""
        CREATE OR REPLACE TABLE {role}_per_game AS
        SELECT
            player_id, game_pk, game_date, n_pitches,
            {per_game_feature_cols()}
        FROM read_parquet('{PITCHER_GAME_PATH}')
        WHERE is_start = {is_start_filter}
    """)

    feature_names = [f"g_{name}_all" for name, _, _ in METRICS]
    for g in PITCH_GROUPS:
        feature_names.append(f"g_pct_{g}")
        for name, _, _ in METRICS:
            feature_names.append(f"g_{name}{g}")
    feature_names.append("n_pitches")

    lag_cols = []
    for lag in range(n_lags):
        for feat in feature_names:
            if lag == 0:
                lag_cols.append(f"{feat} AS lag0_{feat}")
            else:
                lag_cols.append(
                    f"LAG({feat}, {lag}) OVER (PARTITION BY player_id ORDER BY game_date) "
                    f"AS lag{lag}_{feat}"
                )

    con.execute(f"""
        CREATE OR REPLACE TABLE {role}_sequence_raw AS
        SELECT player_id, game_date AS window_end_date,
            {",\n            ".join(lag_cols)}
        FROM {role}_per_game
    """)

    # 라벨/분할/인적사항은 이미 만든 rolling-window 데이터셋에서 그대로 가져다 붙인다.
    window_path = WINDOW_DATASET_DIR / f"{role}_window_dataset.parquet"
    con.execute(f"""
        CREATE OR REPLACE TABLE {role}_sequence AS
        SELECT
            s.*,
            w.p_throws, w.age, w.height_inches, w.weight_lb, w.birth_country,
            w.days_since_prev_game,
            w.il_start_date, w.injury_class_strict, w.days_to_injury, w.label, w.split
        FROM {role}_sequence_raw s
        JOIN read_parquet('{window_path.as_posix()}') w
            ON s.player_id = w.player_id AND s.window_end_date = w.window_end_date
    """)

    out_path = OUT_DIR / f"{role}_sequence_dataset.parquet"
    con.execute(f"COPY {role}_sequence TO '{out_path}' (FORMAT PARQUET)")
    n = con.execute(f"SELECT COUNT(*) FROM {role}_sequence").fetchone()[0]
    n_cols = con.execute(f"SELECT COUNT(*) FROM pragma_table_info('{role}_sequence')").fetchone()[0]
    print(f"[{role}_sequence_dataset] {n:,} rows, {n_cols} columns -> {out_path}")


def main():
    con = duckdb.connect()
    for role in ("bullpen", "starter"):
        build_role(con, role)
    con.close()


if __name__ == "__main__":
    main()
