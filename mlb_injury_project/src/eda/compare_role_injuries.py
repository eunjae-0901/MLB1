"""
4단계: 부상 사건 직전 실제 등판 기록으로 선발/불펜 역할을 구분하고,
역할별 전체/어깨/팔꿈치/그 외 부상 사건 수(고유 episode, 고유 투수)를 비교한다.

역할 판정 기준
  - 각 injury episode의 il_start_date 이전 마지막 등판(같은 날 포함 X, 그 이전)의
    is_start 값을 기본 역할로 삼는다.
  - 다만 IL 등재 직전 30일 이내 등판에 선발/불펜이 섞여 있으면(보직 변경 등)
    "보직변경/혼합"으로 별도 분류한다.
  - il_start_date 이전 등판 기록이 전혀 없으면(데뷔 시즌 전 IL, 수집기간 이전 등판 등)
    "등판기록없음"으로 분류한다.
"""
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PITCHER_GAME_PATH = str(
    PROJECT_ROOT / "data" / "interim" / "pitcher_game" / "pitcher_game_role.parquet"
)
INJURY_PATH = str(
    PROJECT_ROOT / "data" / "interim" / "injury_episodes" / "injury_episodes.parquet"
)
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "role_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_LOOKBACK_DAYS = 30
INJURY_LABEL_MAP = {1: "어깨", 2: "팔꿈치", 3: "그 외"}


def main():
    con = duckdb.connect()

    con.execute(f"""
        CREATE OR REPLACE TABLE injury AS
        SELECT * FROM read_parquet('{INJURY_PATH}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE pg AS
        SELECT * FROM read_parquet('{PITCHER_GAME_PATH}')
    """)

    # 부상 직전 마지막 등판(같은 날 제외, 그 이전) asof 매칭
    con.execute("""
        CREATE OR REPLACE TABLE injury_last_appearance AS
        SELECT
            i.*,
            g.game_date AS last_appearance_date,
            g.is_start AS last_is_start
        FROM injury i
        ASOF LEFT JOIN pg g
            ON i.player_id = g.player_id
           AND g.game_date < i.il_start_date
    """)

    # IL 등재 직전 30일 이내 등판의 역할 다양성(선발/불펜 혼합 여부) 확인
    con.execute(f"""
        CREATE OR REPLACE TABLE injury_role AS
        SELECT
            e.*,
            (SELECT COUNT(DISTINCT g2.is_start) FROM pg g2
             WHERE g2.player_id = e.player_id
               AND g2.game_date < e.il_start_date
               AND g2.game_date >= e.il_start_date - INTERVAL '{ROLE_LOOKBACK_DAYS} days'
            ) AS n_distinct_roles_lookback,
            (SELECT COUNT(*) FROM pg g2
             WHERE g2.player_id = e.player_id
               AND g2.game_date < e.il_start_date
               AND g2.game_date >= e.il_start_date - INTERVAL '{ROLE_LOOKBACK_DAYS} days'
            ) AS n_appearances_lookback,
            CASE
                WHEN last_appearance_date IS NULL THEN '등판기록없음'
                WHEN (SELECT COUNT(DISTINCT g2.is_start) FROM pg g2
                      WHERE g2.player_id = e.player_id
                        AND g2.game_date < e.il_start_date
                        AND g2.game_date >= e.il_start_date - INTERVAL '{ROLE_LOOKBACK_DAYS} days'
                     ) >= 2 THEN '보직변경/혼합'
                WHEN last_is_start THEN '선발'
                ELSE '불펜'
            END AS role_class
        FROM injury_last_appearance e
    """)

    con.execute("""
        CREATE OR REPLACE TABLE injury_role_labeled AS
        SELECT *,
            CASE injury_class_strict
                WHEN 1 THEN '어깨' WHEN 2 THEN '팔꿈치' WHEN 3 THEN '그 외'
                ELSE '알수없음'
            END AS injury_label
        FROM injury_role
    """)

    out_path = OUT_DIR / "injury_role_comparison.parquet"
    con.execute(f"COPY injury_role_labeled TO '{out_path}' (FORMAT PARQUET)")

    print("=== role_class 분포 (고유 injury episode 수) ===")
    print(con.execute("""
        SELECT role_class, COUNT(*) AS n_episodes, COUNT(DISTINCT player_id) AS n_unique_pitchers
        FROM injury_role_labeled GROUP BY 1 ORDER BY 2 DESC
    """).df())

    print("\n=== role_class x 부상유형(Strict) 교차표 (고유 episode 수) ===")
    print(con.execute("""
        SELECT role_class,
               SUM(CASE WHEN injury_label='어깨' THEN 1 ELSE 0 END) AS 어깨,
               SUM(CASE WHEN injury_label='팔꿈치' THEN 1 ELSE 0 END) AS 팔꿈치,
               SUM(CASE WHEN injury_label='그 외' THEN 1 ELSE 0 END) AS 그외,
               COUNT(*) AS 전체
        FROM injury_role_labeled
        GROUP BY role_class
        ORDER BY 전체 DESC
    """).df())

    print("\n=== role_class별 고유 부상 투수 수 (동일 투수가 여러 번 다쳤을 수 있음) ===")
    print(con.execute("""
        SELECT role_class, COUNT(DISTINCT player_id) AS n_unique_injured_pitchers
        FROM injury_role_labeled GROUP BY 1 ORDER BY 2 DESC
    """).df())

    print(f"\n[saved] {out_path}")
    con.close()


if __name__ == "__main__":
    main()
