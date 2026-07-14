"""
직접 눈으로 확인하기 편하도록 parquet 데이터를 Excel(.xlsx) 파일들로 내보낸다.
duckdb + openpyxl만 사용 (pandas/numpy는 이 venv에서 임포트가 깨져있어 의존성에서 제외).
"""
from pathlib import Path

import duckdb
from openpyxl import Workbook

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = PROJECT_ROOT / "data" / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

TABLES = {
    "injury_episodes": PROJECT_ROOT / "data" / "interim" / "injury_episodes" / "injury_episodes.parquet",
    "player_bio": PROJECT_ROOT / "data" / "raw" / "player_bio" / "player_bio.parquet",
    "pitcher_game_role": PROJECT_ROOT / "data" / "interim" / "pitcher_game" / "pitcher_game_role.parquet",
    "role_comparison": PROJECT_ROOT / "data" / "interim" / "role_comparison" / "injury_role_comparison.parquet",
    "bullpen_window_dataset": PROJECT_ROOT / "data" / "processed" / "bullpen_window_dataset.parquet",
    "starter_window_dataset": PROJECT_ROOT / "data" / "processed" / "starter_window_dataset.parquet",
    "bullpen_sequence_dataset": PROJECT_ROOT / "data" / "processed" / "bullpen_sequence_dataset.parquet",
    "starter_sequence_dataset": PROJECT_ROOT / "data" / "processed" / "starter_sequence_dataset.parquet",
}

CHUNK_SIZE = 5000


def _to_cell(value):
    # openpyxl이 못 다루는 타입(예: 그 외 커스텀 객체)만 문자열로 안전 변환
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)


def build_excel(con: duckdb.DuckDBPyConnection):
    for name, path in TABLES.items():
        cur = con.execute(f"SELECT * FROM read_parquet('{path.as_posix()}')")
        cols = [d[0] for d in cur.description]

        wb = Workbook(write_only=True)
        ws = wb.create_sheet(name[:31])
        ws.append(cols)

        n = 0
        while True:
            rows = cur.fetchmany(CHUNK_SIZE)
            if not rows:
                break
            for row in rows:
                ws.append([_to_cell(v) for v in row])
            n += len(rows)

        out_path = EXPORT_DIR / f"{name}.xlsx"
        wb.save(out_path)
        print(f"[excel] {name}: {n:,} rows -> {out_path}")


def main():
    con = duckdb.connect()
    build_excel(con)
    con.close()


if __name__ == "__main__":
    main()
