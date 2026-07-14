"""
투수 기본정보(키, 몸무게, 투구손, 출생국) 수집. MLB Stats API /people 엔드포인트를
배치(300명씩)로 조회한다.

주의(실행 순서, 양방향 의존): 조회 대상 player_id 목록을 03_build_rolling_dataset.py가
만든 data/interim/pitcher_game/pitcher_game_role.parquet에서 가져오므로, 이 스크립트를
돌리려면 03을 먼저 한 번 실행해둬야 한다. 반대로 03은 포지션 필터링(투수가 아닌 선수의
땜빵 등판 제외)에 이 스크립트가 만드는 player_bio.parquet을 사용하므로, 이 스크립트 실행
후에는 03을 다시 한 번 실행해서 최신 player_bio 기준으로 갱신해야 한다.
  순서: 03 실행(1차) -> 04(이 파일) 실행 -> 03 재실행(2차, 최종)

주의: 이 API는 "현재 시점"의 키/몸무게 등 최신 값만 제공한다(시즌별 과거 값 아님).
시즌별로 정밀 추적하기보다 인물 단위 고정값을 쓰되, 추후 시즌별 변화가 중요하면 별도 조정 필요.
"""
import re
import time
from pathlib import Path

import duckdb
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PITCHER_GAME_PATH = str(
    PROJECT_ROOT / "data" / "interim" / "pitcher_game" / "pitcher_game_role.parquet"
)
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "player_bio"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "player_bio.parquet"

BASE_URL = "https://statsapi.mlb.com/api/v1/people"
BATCH_SIZE = 300

HEIGHT_PATTERN = re.compile(r"(\d+)'\s*(\d+)\"?")


def height_to_inches(height_str):
    if not isinstance(height_str, str):
        return None
    m = HEIGHT_PATTERN.search(height_str)
    if not m:
        return None
    feet, inches = int(m.group(1)), int(m.group(2))
    return feet * 12 + inches


def get_unique_pitcher_ids() -> list[int]:
    con = duckdb.connect()
    ids = con.execute(
        f"SELECT DISTINCT player_id FROM read_parquet('{PITCHER_GAME_PATH}')"
    ).df()["player_id"].astype(int).tolist()
    con.close()
    return ids


def fetch_batch(ids: list[int]) -> list[dict]:
    resp = requests.get(
        BASE_URL, params={"personIds": ",".join(str(i) for i in ids)}, timeout=60
    )
    resp.raise_for_status()
    return resp.json().get("people", [])


def main():
    ids = get_unique_pitcher_ids()
    print(f"[bio] 고유 투수 {len(ids):,}명 조회 시작")

    rows = []
    for i in range(0, len(ids), BATCH_SIZE):
        batch = ids[i : i + BATCH_SIZE]
        people = fetch_batch(batch)
        for p in people:
            rows.append(
                {
                    "player_id": p.get("id"),
                    "player_name": p.get("fullName"),
                    "birth_date": p.get("birthDate"),
                    "birth_country": p.get("birthCountry"),
                    "height_raw": p.get("height"),
                    "height_inches": height_to_inches(p.get("height")),
                    "weight_lb": p.get("weight"),
                    "pitch_hand": (p.get("pitchHand") or {}).get("code"),
                    "bat_side": (p.get("batSide") or {}).get("code"),
                    "mlb_debut_date": p.get("mlbDebutDate"),
                    "primary_position_code": (p.get("primaryPosition") or {}).get("code"),
                    "primary_position_name": (p.get("primaryPosition") or {}).get("name"),
                }
            )
        print(f"  {i + len(batch)}/{len(ids)} 완료")
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["mlb_debut_date"] = pd.to_datetime(df["mlb_debut_date"], errors="coerce")
    df.to_parquet(OUT_PATH, index=False)
    print(f"[saved] {OUT_PATH} ({len(df):,} rows)")

    missing = set(ids) - set(df["player_id"])
    if missing:
        print(f"[warn] 조회 실패/누락 player_id {len(missing)}건: {sorted(missing)[:20]}...")


if __name__ == "__main__":
    main()
