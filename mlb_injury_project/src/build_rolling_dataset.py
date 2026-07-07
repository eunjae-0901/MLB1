"""
DuckDB 기반 rolling-window 데이터셋 구축 스켈레톤.

Statcast pitch-level parquet(data/raw/statcast/*.parquet) +
injury_episodes.parquet(data/interim/injury_episodes/)를 이용해
  1) 투수-경기 단위 요약(pitcher_game)
  2) 선발(최근 3경기) / 불펜(최근 14일) rolling 집계
  3) 각 window 이후 PREDICTION_HORIZON_DAYS 이내 IL 등재 여부를 asof join으로 라벨링
까지의 파이프라인을 구현한다.

주의(추후 확정 필요):
  - is_start 판정: boxscore 없이 Statcast만으로 "그 경기, 그 팀에서 가장 먼저 타석
    (at_bat_number)을 상대한 투수가 선발"이라는 규칙으로 판정한다. inning_topbot으로
    투수의 소속팀을 역산(Top이면 홈팀 투수, Bot이면 원정팀 투수)한 뒤, game_pk+team
    단위로 min(at_bat_number)가 가장 작은 투수를 선발로 지정한다. 이전에 쓰던
    "첫 이닝==1회" 근사치보다 정확하며(오프너가 회를 넘겨 던져도 안 흔들림),
    두 방식이 갈리는 경우도 별도로 로그에 남긴다.
  - PREDICTION_HORIZON_DAYS(예측 기간)는 14일로 확정. 불펜은 "최근 14일 -> 다음 14일",
    선발은 "최근 3경기 -> 다음 14일"(3경기를 날짜로 못 박지 않고 불펜과 동일한 14일 캘린더
    기준으로 통일 - "다음 3경기"를 기준으로 삼으면 그 사이 부상으로 등판이 아예 없는 경우
    라벨을 정의할 기준일이 없어지는 문제가 있어 회피함).
  - Rolling-window 데이터 증강(window 시작일을 하루씩 이동)은 의도적으로 아직 미적용.
    나중에 명시적으로 요청받았을 때만 추가할 것.
  - 구종 그룹 매핑(FB/SI/CT/SL/CB/CH/SP)은 Statcast pitch_type 코드 기준 표준 매핑이며,
    스윕컵(ST)·슬러브(SV) 등 최근 신설 구종 분류는 연구 목적에 맞게 재검토 필요.
"""
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATCAST_GLOB = str(PROJECT_ROOT / "data" / "raw" / "statcast" / "statcast_*.parquet")
INJURY_PATH = str(
    PROJECT_ROOT / "data" / "interim" / "injury_episodes" / "injury_episodes.parquet"
)
PLAYER_BIO_PATH = str(
    PROJECT_ROOT / "data" / "raw" / "player_bio" / "player_bio.parquet"
)
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PITCHER_GAME_DIR = PROJECT_ROOT / "data" / "interim" / "pitcher_game"
PITCHER_GAME_DIR.mkdir(parents=True, exist_ok=True)

BULLPEN_WINDOW_DAYS = 14
STARTER_WINDOW_STARTS = 3
PREDICTION_HORIZON_DAYS = 14  # 불펜 14일 window / 선발 3경기 window 모두 "다음 14일" 예측으로 통일

PITCH_GROUP_CASE = """
    CASE pitch_type
        WHEN 'FF' THEN 'FB' WHEN 'FA' THEN 'FB'
        WHEN 'SI' THEN 'SI' WHEN 'FT' THEN 'SI'
        WHEN 'FC' THEN 'CT'
        WHEN 'SL' THEN 'SL' WHEN 'ST' THEN 'SL' WHEN 'SV' THEN 'SL'
        WHEN 'CU' THEN 'CB' WHEN 'KC' THEN 'CB' WHEN 'CS' THEN 'CB'
        WHEN 'CH' THEN 'CH'
        WHEN 'FS' THEN 'SP' WHEN 'FO' THEN 'SP'
        ELSE 'OTHER'
    END
"""
PITCH_GROUPS = ["FB", "SI", "CT", "SL", "CB", "CH", "SP"]

# events -> 아웃 카운트 매핑 (통상적인 타석 결과 기준, 견제사/도루사 등은 반영하지 못하는
# 근사치이다. Statcast pitch-level만으로 IP를 정확히 복원하는 표준 방법은 없어
# 아웃 하나하나를 이벤트에서 역산하는 이 방식이 실무적으로 널리 쓰인다.)
ONE_OUT_EVENTS = [
    "field_out", "strikeout", "force_out", "sac_fly", "sac_bunt",
    "fielders_choice_out",
]
TWO_OUT_EVENTS = [
    "grounded_into_double_play", "double_play", "strikeout_double_play",
    "sac_fly_double_play", "sac_bunt_double_play",
]
THREE_OUT_EVENTS = ["triple_play"]

OUTS_CASE = f"""
    CASE
        WHEN events IN ({",".join(repr(e) for e in ONE_OUT_EVENTS)}) THEN 1
        WHEN events IN ({",".join(repr(e) for e in TWO_OUT_EVENTS)}) THEN 2
        WHEN events IN ({",".join(repr(e) for e in THREE_OUT_EVENTS)}) THEN 3
        ELSE 0
    END
"""


def group_cols_sql() -> str:
    """구종 그룹별 n/속도합/수평합/수직합/익스텐션합/회전수합 컬럼 생성 SQL."""
    parts = []
    for g in PITCH_GROUPS:
        parts.append(f"""
        SUM(CASE WHEN pitch_group = '{g}' THEN 1 ELSE 0 END) AS n_{g},
        SUM(CASE WHEN pitch_group = '{g}' THEN release_speed ELSE 0 END) AS sum_v{g},
        SUM(CASE WHEN pitch_group = '{g}' THEN pfx_x * 12 ELSE 0 END) AS sum_x{g},
        SUM(CASE WHEN pitch_group = '{g}' THEN pfx_z * 12 ELSE 0 END) AS sum_z{g},
        SUM(CASE WHEN pitch_group = '{g}' THEN release_extension ELSE 0 END) AS sum_ext{g},
        SUM(CASE WHEN pitch_group = '{g}' THEN release_spin_rate ELSE 0 END) AS sum_spin{g}
        """)
    return ",\n".join(parts)


def build_pitcher_game(con: duckdb.DuckDBPyConnection):
    con.execute(f"""
        CREATE OR REPLACE TABLE pitcher_game AS
        WITH pitches AS (
            SELECT
                pitcher AS player_id,
                game_pk,
                CAST(game_date AS DATE) AS game_date,
                p_throws,
                age_pit AS age,
                inning,
                at_bat_number,
                events,
                CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END
                    AS pitcher_team,
                release_speed, pfx_x, pfx_z, release_extension, release_spin_rate,
                {PITCH_GROUP_CASE} AS pitch_group
            FROM read_parquet('{STATCAST_GLOB}')
            WHERE game_type = 'R'  -- 정규시즌만 (스프링캠프/포스트시즌 제외, 추후 조정 가능)
        )
        SELECT
            player_id, game_pk, game_date,
            ANY_VALUE(pitcher_team) AS pitcher_team,
            ANY_VALUE(p_throws) AS p_throws,
            MIN(age) AS age,
            MIN(inning) AS first_inning,
            MIN(at_bat_number) AS first_ab_number,
            COUNT(*) AS n_pitches,
            COUNT(DISTINCT at_bat_number) AS n_batters_faced,
            SUM({OUTS_CASE}) AS outs_recorded,
            SUM(release_speed) AS sum_v_all,
            SUM(pfx_x * 12) AS sum_x_all,
            SUM(pfx_z * 12) AS sum_z_all,
            SUM(release_extension) AS sum_ext_all,
            SUM(release_spin_rate) AS sum_spin_all,
            {group_cols_sql()}
        FROM pitches
        GROUP BY player_id, game_pk, game_date
    """)
    n = con.execute("SELECT COUNT(*) FROM pitcher_game").fetchone()[0]
    print(f"[pitcher_game] {n:,} rows")


def build_role_flag(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE OR REPLACE TABLE pitcher_game_role AS
        SELECT
            pg.*,
            (ROW_NUMBER() OVER (
                PARTITION BY pg.game_pk, pg.pitcher_team ORDER BY pg.first_ab_number
            ) = 1) AS is_start,
            (pg.first_inning = 1) AS is_start_heuristic_v1,
            (t.n_pitchers = 1) AS is_complete_game,
            CAST(t.n_pitchers = 1 AS INTEGER) AS complete_game_flag
        FROM pitcher_game pg
        JOIN (
            SELECT game_pk, pitcher_team, COUNT(DISTINCT player_id) AS n_pitchers
            FROM pitcher_game GROUP BY 1, 2
        ) t ON pg.game_pk = t.game_pk AND pg.pitcher_team = t.pitcher_team
    """)
    counts = con.execute(
        "SELECT is_start, COUNT(*) FROM pitcher_game_role GROUP BY 1"
    ).df()
    print("[role split (at_bat_number 기준)]\n", counts)

    mismatch = con.execute("""
        SELECT COUNT(*) FROM pitcher_game_role
        WHERE is_start != is_start_heuristic_v1
    """).fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM pitcher_game_role").fetchone()[0]
    print(f"[v1 첫이닝 근사치와 판정이 갈린 경기 수] {mismatch:,} / {total:,}")

    out_path = PITCHER_GAME_DIR / "pitcher_game_role.parquet"
    con.execute(f"COPY pitcher_game_role TO '{out_path}' (FORMAT PARQUET)")
    print(f"[saved] {out_path}")


def _rolling_select(frame_clause: str) -> str:
    """rolling 누적합 컬럼 생성 SQL (starter/bullpen 공용)."""
    agg_cols = ["n_pitches", "n_batters_faced", "outs_recorded", "complete_game_flag",
                "sum_v_all", "sum_x_all", "sum_z_all", "sum_ext_all", "sum_spin_all"]
    for g in PITCH_GROUPS:
        agg_cols += [f"n_{g}", f"sum_v{g}", f"sum_x{g}", f"sum_z{g}",
                     f"sum_ext{g}", f"sum_spin{g}"]
    lines = [
        f"SUM({c}) OVER (PARTITION BY player_id ORDER BY game_date {frame_clause}) "
        f"AS roll_{c}"
        for c in agg_cols
    ]
    lines.append(
        "COUNT(*) OVER (PARTITION BY player_id ORDER BY game_date "
        f"{frame_clause}) AS roll_n_appearances"
    )
    return ",\n        ".join(lines)


def build_rolling_windows(con: duckdb.DuckDBPyConnection):
    # 불펜: 최근 14일 (날짜 기준 RANGE 프레임)
    con.execute(f"""
        CREATE OR REPLACE TABLE bullpen_rolling AS
        SELECT
            player_id, game_pk, game_date, p_throws, age,
            {_rolling_select(f"RANGE BETWEEN INTERVAL {BULLPEN_WINDOW_DAYS - 1} DAYS PRECEDING AND CURRENT ROW")}
        FROM pitcher_game_role
        WHERE is_start = false
    """)
    # 선발: 최근 3경기 (행 기준 ROWS 프레임)
    con.execute(f"""
        CREATE OR REPLACE TABLE starter_rolling AS
        SELECT
            player_id, game_pk, game_date, p_throws, age,
            {_rolling_select(f"ROWS BETWEEN {STARTER_WINDOW_STARTS - 1} PRECEDING AND CURRENT ROW")}
        FROM pitcher_game_role
        WHERE is_start = true
    """)
    for t in ("bullpen_rolling", "starter_rolling"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"[{t}] {n:,} rows")


def finalize_averages(con: duckdb.DuckDBPyConnection, table: str, role: str):
    """rolling 누적합 -> 실제 평균/비율로 변환 + injury label asof join."""
    avg_cols = ["""
        roll_sum_v_all / NULLIF(roll_n_pitches, 0) AS w_v_all,
        roll_sum_x_all / NULLIF(roll_n_pitches, 0) AS w_x_all,
        roll_sum_z_all / NULLIF(roll_n_pitches, 0) AS w_z_all,
        roll_sum_ext_all / NULLIF(roll_n_pitches, 0) AS w_ext_all,
        roll_sum_spin_all / NULLIF(roll_n_pitches, 0) AS w_spin_all
    """]
    for g in PITCH_GROUPS:
        avg_cols.append(f"""
        roll_n_{g} / NULLIF(roll_n_pitches, 0) AS w_pct_{g},
        roll_sum_v{g} / NULLIF(roll_n_{g}, 0) AS w_v{g},
        roll_sum_x{g} / NULLIF(roll_n_{g}, 0) AS w_x{g},
        roll_sum_z{g} / NULLIF(roll_n_{g}, 0) AS w_z{g},
        roll_sum_ext{g} / NULLIF(roll_n_{g}, 0) AS w_ext{g},
        roll_sum_spin{g} / NULLIF(roll_n_{g}, 0) AS w_spin{g}
        """)

    con.execute(f"""
        CREATE OR REPLACE TABLE {table}_features AS
        SELECT
            b.player_id, b.game_date AS window_end_date, b.p_throws, b.age,
            b.roll_n_pitches AS n_pitches_window,
            b.roll_n_batters_faced AS n_batters_faced_window,
            b.roll_n_appearances AS n_appearances_window,
            b.roll_outs_recorded / 3.0 AS innings_pitched_window,
            b.roll_complete_game_flag AS complete_games_window,
            bio.height_inches, bio.weight_lb, bio.birth_country,
            {",".join(avg_cols)}
        FROM {table} b
        LEFT JOIN read_parquet('{PLAYER_BIO_PATH}') bio ON b.player_id = bio.player_id
    """)

    # asof join: window 종료일 이후 가장 가까운 IL 등재 사건 연결
    con.execute(f"""
        CREATE OR REPLACE TABLE {table}_labeled AS
        SELECT
            f.*,
            i.il_start_date,
            i.injury_class_strict,
            date_diff('day', f.window_end_date, i.il_start_date) AS days_to_injury,
            CASE
                WHEN i.il_start_date IS NOT NULL
                     AND date_diff('day', f.window_end_date, i.il_start_date)
                         BETWEEN 0 AND {PREDICTION_HORIZON_DAYS}
                THEN i.injury_class_strict
                ELSE 0
            END AS label
        FROM {table}_features f
        ASOF LEFT JOIN read_parquet('{INJURY_PATH}') i
            ON f.player_id = i.player_id
           AND f.window_end_date <= i.il_start_date
    """)

    out_path = OUT_DIR / f"{role}_window_dataset.parquet"
    con.execute(f"COPY {table}_labeled TO '{out_path}' (FORMAT PARQUET)")
    label_dist = con.execute(
        f"SELECT label, COUNT(*) FROM {table}_labeled GROUP BY 1 ORDER BY 1"
    ).df()
    print(f"[saved] {out_path}")
    print(label_dist)


def main():
    con = duckdb.connect()
    build_pitcher_game(con)
    build_role_flag(con)
    build_rolling_windows(con)
    finalize_averages(con, "bullpen_rolling", "bullpen")
    finalize_averages(con, "starter_rolling", "starter")
    con.close()


if __name__ == "__main__":
    main()
