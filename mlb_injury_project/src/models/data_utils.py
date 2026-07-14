"""01_xgboost_bayesian.py, 02_dnn_bayesian.py가 공유하는 rolling-window 데이터 로딩."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ID_COLS = ["player_id", "window_end_date", "il_start_date", "injury_class_strict", "days_to_injury", "split"]
CATEGORICAL_COLS = ["p_throws", "birth_country"]


def load_role(role: str, exclude_other: bool = True, binarize: bool = False) -> dict[str, pd.DataFrame]:
    """{role}_window_dataset.parquet을 읽어 train/val/test로 나눠 반환한다.

    exclude_other=True(기본값)면 label==3('그 외') 행을 제거한다 - '그 외'는
    허리/무릎/질병 등 서로 무관한 부상을 뭉뚱그린 잡동사니 카테고리라 어깨·팔꿈치
    예측에 방해가 될 수 있다는 판단.

    binarize=False(기본값)면 남은 라벨을 0(안다침)/1(어깨)/2(팔꿈치) 3종 분류로 두고,
    binarize=True면 1과 2를 합쳐서 0(안다침) vs 1(어깨 또는 팔꿈치) 이진분류로 만든다.
    """
    path = PROCESSED_DIR / f"{role}_window_dataset.parquet"
    df = pd.read_parquet(path)
    if exclude_other:
        df = df[df["label"] != 3].copy()
    if binarize:
        df["label"] = (df["label"] > 0).astype(int)

    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")

    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}


def numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + CATEGORICAL_COLS + ["label"]]
