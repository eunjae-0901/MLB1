"""
pybaseball을 이용해 연도별 Statcast pitch-level 데이터를 수집해 parquet로 저장한다.
시즌 범위는 스프링캠프 후반~월드시리즈까지 넉넉히 잡아 결측 없이 커버한다.
"""
import time
from pathlib import Path

import pandas as pd
import pybaseball as pb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "statcast"
RAW_DIR.mkdir(parents=True, exist_ok=True)

pb.cache.enable()

START_YEAR, END_YEAR = 2016, 2025


def season_range(year: int) -> tuple[str, str]:
    return f"{year}-03-25", f"{year}-11-05"


def main():
    for year in range(START_YEAR, END_YEAR + 1):
        out_path = RAW_DIR / f"statcast_{year}.parquet"
        if out_path.exists():
            print(f"[skip] {year} already collected -> {out_path}")
            continue

        start_dt, end_dt = season_range(year)
        print(f"[fetch] {year} ({start_dt} ~ {end_dt}) ...", flush=True)
        t0 = time.time()
        df = pb.statcast(start_dt=start_dt, end_dt=end_dt, verbose=False)
        elapsed = time.time() - t0
        print(f"  rows={len(df):,}  elapsed={elapsed/60:.1f}min", flush=True)

        # 투수 부상 예측에 필요 없는 초대형/중복 텍스트 컬럼은 유지하되 저장은 그대로 parquet에
        df.to_parquet(out_path, index=False)
        print(f"[saved] {out_path}", flush=True)
        time.sleep(3)


if __name__ == "__main__":
    main()
