"""
DuckDB 기반 rolling-window 데이터셋 구축 스켈레톤.

Statcast pitch-level parquet(data/raw/statcast/*.parquet) +
injury_episodes.parquet(data/interim/injury_episodes/)를 이용해
  1) 투수-경기 단위 요약(pitcher_game)
  2) 선발(최근 3경기) / 불펜(최근 14일) rolling 집계
  3) 각 window 이후 PREDICTION_HORIZON_DAYS 이내 IL 등재 여부를 asof join으로 라벨링
까지의 파이프라인을 구현한다.

주의(실행 순서, 04_collect_player_bio.py와 양방향 의존): 포지션 필터링(투수가 아닌 선수의
땜빵 등판 제외)에 data/raw/player_bio/player_bio.parquet이 필요한데, 이건 04번 스크립트가
만든다. 그런데 04번은 이 스크립트가 만드는 pitcher_game_role.parquet에서 조회 대상
player_id 목록을 가져온다. 그래서 최초 세팅 시 이 스크립트를 먼저 한 번 실행 -> 04 실행
-> 이 스크립트를 다시 실행(최종, player_bio 반영) 순서로 두 번 돌려야 한다.

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
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_defs import METRICS, PITCH_GROUPS  # noqa: E402

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

# 항목 4-1(다기간 workload 변수): 기존 14일 하나만으로는 "최근에 몰아서 던졌는지"를
# 구분 못 해서, 더 짧은 기간의 등판수/투구수를 추가로 집계한다(불펜 전용 - 선발은
# 등판 간격이 며칠 단위라 3/7일 구간 개념이 안 맞아 경기 단위 변수를 따로 둔다).
BULLPEN_WORKLOAD_SHORT_DAYS = 3
BULLPEN_WORKLOAD_MED_DAYS = 7

# 선발의 "최근 3경기"는 횟수(ROWS) 기준이라 시즌 경계를 모르고 그냥 이어붙인다. 그래서
# 새 시즌 첫 등판은 작년 시즌 마지막 등판(몇 달 전)이 3경기 평균에 섞여버리는 문제가 있었다.
# 직전 등판과 이 값(일) 넘게 벌어지면 오프시즌을 건넌 것으로 보고 그 이전 등판은 아예
# rolling 대상에서 끊는다(불펜은 14일 RANGE 프레임이라 오프시즌 공백(최소 61일 확인됨)을
# 절대 못 넘어와서 이 문제 자체가 없다).
STARTER_SEASON_GAP_DAYS = 20

# 시즌 마지막 날짜에서 이 값(일) 이내로 끝나는 window는 "다음 PREDICTION_HORIZON_DAYS일
# 이내 부상 여부"를 확인할 방법이 사실상 없다(시즌 끝나면 IL 등재 자체가 다음해 스프링캠프로
# 밀리는 경우가 많아, 실측 결과 시즌 종료 직후 14일간 리그 전체 IL 등재가 연 0~3건에 불과).
# 그래서 "확인 불가능한데 라벨만 0으로 찍히는" 것을 막기 위해 이 구간의 window는 제외한다.
SEASON_END_BUFFER_DAYS = PREDICTION_HORIZON_DAYS

# train/val/test 6:2:2 시간 기준 분할 (2016~2025 10시즌 -> 6/2/2년).
# val·test 시작 직후 BUFFER일 동안의 window는 그 이전 split의 투구 데이터를 일부
# 포함할 수 있어(rolling window가 과거로 걸쳐있으므로) 아예 제외한다. 선발은 window가
# "3경기"라 실제 날짜 폭이 늘어질 수 있어(올스타 브레이크 등) 더 넉넉하게 버퍼를 둔다.
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
SPLIT_BUFFER_DAYS = {"bullpen": BULLPEN_WINDOW_DAYS, "starter": 30}

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
    """구종 그룹별 n(전체 투구수, 사용비율 계산용)/각 지표 합·유효개수 컬럼 생성 SQL."""
    parts = []
    for g in PITCH_GROUPS:
        cols = [f"SUM(CASE WHEN pitch_group = '{g}' THEN 1 ELSE 0 END) AS n_{g}"]
        for name, sum_expr, null_col in METRICS:
            cols.append(
                f"SUM(CASE WHEN pitch_group = '{g}' THEN {sum_expr} ELSE 0 END) "
                f"AS sum_{name}{g}"
            )
            cols.append(
                f"SUM(CASE WHEN pitch_group = '{g}' AND {null_col} IS NOT NULL "
                f"THEN 1 ELSE 0 END) AS n{name}{g}"
            )
        parts.append(",\n        ".join(cols))
    return ",\n".join(parts)


def all_metric_cols_sql() -> str:
    """전체(구종 무관) 지표 합·유효개수 컬럼 생성 SQL."""
    cols = []
    for name, sum_expr, null_col in METRICS:
        cols.append(f"SUM({sum_expr}) AS sum_{name}_all")
        cols.append(f"COUNT({null_col}) AS n{name}_all")
    return ",\n            ".join(cols)


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
                pitcher_days_since_prev_game,
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
            MIN(pitcher_days_since_prev_game) AS days_since_prev_game,
            COUNT(*) AS n_pitches,
            COUNT(DISTINCT at_bat_number) AS n_batters_faced,
            SUM({OUTS_CASE}) AS outs_recorded,
            {all_metric_cols_sql()},
            {group_cols_sql()}
        FROM pitches
        GROUP BY player_id, game_pk, game_date
    """)
    n = con.execute("SELECT COUNT(*) FROM pitcher_game").fetchone()[0]
    print(f"[pitcher_game] {n:,} rows")


def build_role_flag(con: duckdb.DuckDBPyConnection):
    # 야수가 대량득실차 경기에 땜빵으로 등판한 기록 제외. primary_position_code가
    # 'Y'(투타겸업, 예: 오타니)나 '1'(투수)이 아니면 실제 투수가 아닌 것으로 판정.
    # bio 조회가 안 된 선수는 보수적으로 그대로 둔다(결측이라고 제외하지 않음).
    excluded = con.execute(f"""
        SELECT COUNT(*) FROM pitcher_game pg
        JOIN read_parquet('{PLAYER_BIO_PATH}') bio ON pg.player_id = bio.player_id
        WHERE bio.primary_position_code NOT IN ('1', 'Y')
    """).fetchone()[0]
    print(f"[야수 등판 제외] {excluded:,}건")

    con.execute(f"""
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
        LEFT JOIN read_parquet('{PLAYER_BIO_PATH}') bio ON pg.player_id = bio.player_id
        WHERE bio.primary_position_code IS NULL
           OR bio.primary_position_code IN ('1', 'Y')
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


def _rolling_select(frame_clause: str, partition_cols: str = "player_id") -> str:
    """rolling 누적합 컬럼 생성 SQL (starter/bullpen 공용).
    partition_cols: 선발은 시즌 경계(season_group)까지 같이 파티션해서 오프시즌
    너머 등판이 섞이지 않게 한다."""
    agg_cols = ["n_pitches", "n_batters_faced", "outs_recorded", "complete_game_flag"]
    for name, _, _ in METRICS:
        agg_cols += [f"sum_{name}_all", f"n{name}_all"]
    for g in PITCH_GROUPS:
        agg_cols.append(f"n_{g}")
        for name, _, _ in METRICS:
            agg_cols += [f"sum_{name}{g}", f"n{name}{g}"]
    lines = [
        f"SUM({c}) OVER (PARTITION BY {partition_cols} ORDER BY game_date {frame_clause}) "
        f"AS roll_{c}"
        for c in agg_cols
    ]
    lines.append(
        f"COUNT(*) OVER (PARTITION BY {partition_cols} ORDER BY game_date "
        f"{frame_clause}) AS roll_n_appearances"
    )
    return ",\n        ".join(lines)


def _workload_only_select(frame_clause: str, suffix: str, partition_cols: str = "player_id") -> str:
    """항목 4-1용: 등판수/투구수만 짧은 기간(3일/7일)으로 따로 집계한다.
    전체 56개 지표를 다 3중으로 늘리면 컬럼이 폭증하니, 여기선 workload 관련
    두 개만 뽑는다."""
    return (
        f"SUM(n_pitches) OVER (PARTITION BY {partition_cols} ORDER BY game_date {frame_clause}) "
        f"AS roll_n_pitches_{suffix},\n        "
        f"COUNT(*) OVER (PARTITION BY {partition_cols} ORDER BY game_date {frame_clause}) "
        f"AS roll_n_appearances_{suffix}"
    )


def build_rolling_windows(con: duckdb.DuckDBPyConnection):
    # 불펜: 최근 14일 (날짜 기준 RANGE 프레임). 오프시즌 공백은 항상 14일보다 훨씬 길어서
    # (실측 최소 61일) 시즌 경계를 따로 안 챙겨도 작년 데이터가 섞일 수 없다.
    #
    # 여기에 항목 4-1(다기간 workload/휴식) 변수를 추가한다:
    #   - 3일/7일 짧은 기간 등판수·투구수 (14일 하나만으론 "최근에 몰아 던졌는지" 구분 불가)
    #   - 최근 14일 등판 간격의 평균/최솟값: days_since_prev_game 자체가 "그 등판 시점의
    #     직전 간격"이라, 14일 RANGE 프레임으로 AVG/MIN을 내면 그대로 "최근 등판들의
    #     간격 평균/최솟값"이 된다(새로 복잡한 배열 계산 안 해도 됨).
    #   - 연투 여부(back_to_back): 직전 간격이 1일 이하
    #   - 3일 연속 등판 여부(three_straight): 이번 등판과 그 직전 등판이 둘 다 간격 1일 이하
    #     (예: 날짜 D, D-1, D-2에 다 등판)
    con.execute(f"""
        CREATE OR REPLACE TABLE bullpen_rolling AS
        SELECT
            player_id, game_pk, game_date, p_throws, age, days_since_prev_game,
            {_rolling_select(f"RANGE BETWEEN INTERVAL {BULLPEN_WINDOW_DAYS - 1} DAYS PRECEDING AND CURRENT ROW")},
            {_workload_only_select(f"RANGE BETWEEN INTERVAL {BULLPEN_WORKLOAD_SHORT_DAYS - 1} DAYS PRECEDING AND CURRENT ROW", "3d")},
            {_workload_only_select(f"RANGE BETWEEN INTERVAL {BULLPEN_WORKLOAD_MED_DAYS - 1} DAYS PRECEDING AND CURRENT ROW", "7d")},
            AVG(days_since_prev_game) OVER (
                PARTITION BY player_id ORDER BY game_date
                RANGE BETWEEN INTERVAL {BULLPEN_WINDOW_DAYS - 1} DAYS PRECEDING AND CURRENT ROW
            ) AS roll_avg_rest_days_14d,
            MIN(days_since_prev_game) OVER (
                PARTITION BY player_id ORDER BY game_date
                RANGE BETWEEN INTERVAL {BULLPEN_WINDOW_DAYS - 1} DAYS PRECEDING AND CURRENT ROW
            ) AS roll_min_rest_days_14d,
            CAST(COALESCE(days_since_prev_game <= 1, false) AS INTEGER) AS back_to_back_flag,
            CAST(COALESCE(
                days_since_prev_game <= 1
                AND LAG(days_since_prev_game) OVER (PARTITION BY player_id ORDER BY game_date) <= 1,
                false
            ) AS INTEGER) AS three_straight_flag
        FROM pitcher_game_role
        WHERE is_start = false
    """)

    # 선발: 최근 3경기 (행 기준 ROWS 프레임). ROWS 프레임은 시즌 경계를 모르고 그냥
    # 이어붙이므로, 직전 등판과 STARTER_SEASON_GAP_DAYS일 넘게 벌어진 지점마다
    # season_group을 새로 매겨서(누적 카운트) 그 이전 시즌 등판이 3경기 평균에
    # 안 섞이도록 파티션을 끊는다.
    con.execute(f"""
        CREATE OR REPLACE TABLE starter_with_group AS
        WITH gapped AS (
            SELECT *,
                game_date - LAG(game_date) OVER (PARTITION BY player_id ORDER BY game_date)
                    AS gap_days
            FROM pitcher_game_role
            WHERE is_start = true
        )
        SELECT *,
            SUM(CASE WHEN gap_days IS NULL OR gap_days > {STARTER_SEASON_GAP_DAYS}
                     THEN 1 ELSE 0 END)
                OVER (PARTITION BY player_id ORDER BY game_date ROWS UNBOUNDED PRECEDING)
                AS season_group
        FROM gapped
    """)
    # 선발용 항목 4-1 변수: 불펜과 달리 등판 간격이 며칠 단위라 3/7일 구간 개념이
    # 안 맞아서, 경기 단위(직전 경기/최근 2경기/최근 3경기 최대치)로 대신한다.
    con.execute(f"""
        CREATE OR REPLACE TABLE starter_rolling AS
        SELECT
            player_id, game_pk, game_date, p_throws, age, days_since_prev_game,
            {_rolling_select(
                f"ROWS BETWEEN {STARTER_WINDOW_STARTS - 1} PRECEDING AND CURRENT ROW",
                "player_id, season_group",
            )},
            n_pitches AS last_game_pitches,
            n_pitches + COALESCE(
                LAG(n_pitches) OVER (PARTITION BY player_id, season_group ORDER BY game_date), 0
            ) AS last_2game_pitches,
            MAX(n_pitches) OVER (
                PARTITION BY player_id, season_group ORDER BY game_date
                ROWS BETWEEN {STARTER_WINDOW_STARTS - 1} PRECEDING AND CURRENT ROW
            ) AS max_pitches_3g,
            CAST(COALESCE(days_since_prev_game <= 4, false) AS INTEGER) AS short_rest_flag
        FROM starter_with_group
    """)
    for t in ("bullpen_rolling", "starter_rolling"):
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"[{t}] {n:,} rows")


def finalize_averages(con: duckdb.DuckDBPyConnection, table: str, role: str):
    """rolling 누적합 -> 실제 평균/비율로 변환 + injury label asof join."""
    # 주의: 평균은 roll_n_pitches(전체 투구수)가 아니라 각 지표별 유효(비결측) 투구수로
    # 나눈다. 일부 경기는 트래킹 장비 오류로 구속/무브먼트 등이 통째로 결측인 경우가 있어,
    # 전체 투구수로 나누면 평균이 실제보다 훨씬 낮게 왜곡된다.
    avg_cols = [
        ",\n        ".join(
            f"roll_sum_{name}_all / NULLIF(roll_n{name}_all, 0) AS w_{name}_all"
            for name, _, _ in METRICS
        )
    ]
    for g in PITCH_GROUPS:
        lines = [f"roll_n_{g} / NULLIF(roll_n_pitches, 0) AS w_pct_{g}"]
        for name, _, _ in METRICS:
            lines.append(
                f"roll_sum_{name}{g} / NULLIF(roll_n{name}{g}, 0) AS w_{name}{g}"
            )
        avg_cols.append(",\n        ".join(lines))

    # 항목 4-1(다기간 workload/휴식) 변수 - 역할별로 이름이 다르다(불펜은 03_build_
    # rolling_dataset.py의 build_rolling_windows에서 만든 3일/7일/휴식일 집계,
    # 선발은 경기 단위 집계).
    if role == "bullpen":
        workload_cols = """
            b.roll_n_pitches_3d AS n_pitches_3d, b.roll_n_appearances_3d AS n_appearances_3d,
            b.roll_n_pitches_7d AS n_pitches_7d, b.roll_n_appearances_7d AS n_appearances_7d,
            b.roll_avg_rest_days_14d AS avg_rest_days_14d,
            b.roll_min_rest_days_14d AS min_rest_days_14d,
            b.back_to_back_flag, b.three_straight_flag,
        """
    else:
        workload_cols = """
            b.last_game_pitches, b.last_2game_pitches, b.max_pitches_3g, b.short_rest_flag,
        """

    con.execute(f"""
        CREATE OR REPLACE TABLE {table}_features AS
        SELECT
            b.player_id, b.game_date AS window_end_date, b.p_throws, b.age,
            b.days_since_prev_game,
            b.roll_n_pitches AS n_pitches_window,
            b.roll_n_batters_faced AS n_batters_faced_window,
            b.roll_n_appearances AS n_appearances_window,
            b.roll_outs_recorded / 3.0 AS innings_pitched_window,
            b.roll_complete_game_flag AS complete_games_window,
            {workload_cols}
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

    # 시즌 마지막 날짜(그 해 실제 정규시즌 마지막 경기일) 확인용 테이블. 이 날짜에서
    # SEASON_END_BUFFER_DAYS 이내로 끝나는 window는 "다음 PREDICTION_HORIZON_DAYS일
    # 이내 부상 여부"를 관측할 수가 없어서(시즌 끝나면 IL 등재가 몇 달 뒤로 밀림) 제외한다.
    con.execute("""
        CREATE OR REPLACE TABLE season_ends AS
        SELECT EXTRACT(year FROM game_date) AS yr, MAX(game_date) AS season_end_date
        FROM pitcher_game_role
        GROUP BY 1
    """)

    buffer_days = SPLIT_BUFFER_DAYS[role]
    con.execute(f"""
        CREATE OR REPLACE TABLE {table}_split AS
        SELECT l.*,
            CASE
                WHEN l.window_end_date > se.season_end_date - INTERVAL '{SEASON_END_BUFFER_DAYS} days'
                    THEN 'dropped_season_end'
                WHEN l.window_end_date < DATE '{TRAIN_END}' THEN 'train'
                WHEN l.window_end_date >= DATE '{VAL_START}' + INTERVAL '{buffer_days} days'
                     AND l.window_end_date < DATE '{VAL_END}' THEN 'val'
                WHEN l.window_end_date >= DATE '{TEST_START}' + INTERVAL '{buffer_days} days'
                    THEN 'test'
                ELSE 'dropped_buffer'
            END AS split
        FROM {table}_labeled l
        JOIN season_ends se ON EXTRACT(year FROM l.window_end_date) = se.yr
    """)

    dropped = con.execute(
        f"SELECT COUNT(*) FROM {table}_split WHERE split = 'dropped_buffer'"
    ).fetchone()[0]
    print(f"[split buffer 제외] {dropped:,}건 (경계 겹침 방지용, buffer={buffer_days}일)")
    dropped_season_end = con.execute(
        f"SELECT COUNT(*) FROM {table}_split WHERE split = 'dropped_season_end'"
    ).fetchone()[0]
    print(f"[시즌 종료 임박 제외] {dropped_season_end:,}건 (관측 불가 구간, buffer={SEASON_END_BUFFER_DAYS}일)")

    out_path = OUT_DIR / f"{role}_window_dataset.parquet"
    con.execute(f"""
        COPY (SELECT * FROM {table}_split WHERE split NOT IN ('dropped_buffer', 'dropped_season_end'))
        TO '{out_path}' (FORMAT PARQUET)
    """)
    split_dist = con.execute(f"""
        SELECT split, label, COUNT(*) AS n
        FROM {table}_split
        WHERE split NOT IN ('dropped_buffer', 'dropped_season_end')
        GROUP BY 1, 2 ORDER BY 1, 2
    """).df()
    print(f"[saved] {out_path}")
    print(split_dist.pivot(index="split", columns="label", values="n"))


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
