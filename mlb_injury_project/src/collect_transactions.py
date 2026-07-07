"""
MLB Stats API에서 연도별 transaction 데이터를 수집해 parquet로 저장한다.
문서(2단계 데이터 수집 및 데이터셋 구축) 1단계 사양을 그대로 재현.
"""
import time
from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "transactions"
RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://statsapi.mlb.com/api/v1/transactions"


def fetch_year(year: int) -> pd.DataFrame:
    resp = requests.get(
        BASE_URL,
        params={
            "startDate": f"{year}-01-01",
            "endDate": f"{year}-12-31",
            "sportId": 1,  # MLB(메이저리그)만 포함, 마이너/아카데미 레벨 제외
        },
        timeout=60,
    )
    resp.raise_for_status()
    txns = resp.json().get("transactions", [])

    rows = []
    for t in txns:
        person = t.get("person") or {}
        rows.append(
            {
                "query_year": year,
                "transaction_id": t.get("id"),
                "player_id": person.get("id"),
                "player_name": person.get("fullName"),
                "date": t.get("date"),
                "effective_date": t.get("effectiveDate"),
                "resolution_date": t.get("resolutionDate"),
                "type_code": t.get("typeCode"),
                "type_desc": t.get("typeDesc"),
                "description_raw": t.get("description"),
                "from_team": (t.get("fromTeam") or {}).get("name"),
                "to_team": (t.get("toTeam") or {}).get("name"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
        df["resolution_date"] = pd.to_datetime(df["resolution_date"], errors="coerce")
    return df


def main(start_year: int = 2015, end_year: int = 2025):
    for year in range(start_year, end_year + 1):
        out_path = RAW_DIR / f"transactions_{year}.parquet"
        if out_path.exists():
            print(f"[skip] {year} already collected -> {out_path}")
            continue
        print(f"[fetch] {year} ...")
        df = fetch_year(year)
        df.to_parquet(out_path, index=False)
        print(f"[saved] {year}: {len(df):,} rows -> {out_path}")
        time.sleep(1)  # MLB Stats API 예의상 딜레이


if __name__ == "__main__":
    main()
