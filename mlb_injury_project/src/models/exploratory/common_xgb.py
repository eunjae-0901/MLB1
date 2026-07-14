"""XGBoost 계열(01, 02) 스크립트가 공유하는 데이터 로딩/평가 유틸."""
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLS = [
    "player_id", "window_end_date", "il_start_date",
    "injury_class_strict", "days_to_injury", "split", "label",
]
CATEGORICAL_COLS = ["p_throws", "birth_country"]


def load_role(role: str, exclude_other: bool = False, binarize: bool = True,
              merge_other_into_healthy: bool = False):
    """role별 window_dataset.parquet을 읽어 train/val/test로 나눠 반환.

    exclude_other=True면 label==3('그 외') 행을 아예 제거한다(표본 수 자체가 줄어듦).
    binarize=True(기본값)면 남은 라벨을 이진(0=안다침 / 1=어깨·팔꿈치)으로 재라벨링하고,
    binarize=False면 0/1/2(안다침/어깨/팔꿈치) 3종 분류로 그대로 둔다.

    merge_other_into_healthy=True면 exclude_other와 달리 행을 하나도 안 지우고,
    label==3을 0으로 재라벨링만 한다("그 외 부상"도 "어깨·팔꿈치는 안 다쳤다"는
    정보로 취급 - 표본 수는 그대로 유지하면서 0/1/2 3종 분류를 한다).
    """
    path = PROCESSED_DIR / f"{role}_window_dataset.parquet"
    df = pd.read_parquet(path)

    if merge_other_into_healthy:
        df = df.copy()
        df["label"] = df["label"].replace(3, 0)
    elif exclude_other:
        df = df[df["label"] != 3].copy()
        if binarize:
            df["label"] = (df["label"] > 0).astype(int)

    # train/val/test로 나누기 전에 범주형 category 목록을 전체 기준으로 고정
    # (그렇지 않으면 특정 값이 train엔 없고 val/test에만 있을 때 XGBoost가 에러를 낸다)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")

    return {s: df[df["split"] == s] for s in ("train", "val", "test")}


def prepare_xy(df: pd.DataFrame):
    X = df.drop(columns=DROP_COLS).copy()
    y = df["label"].astype(int)
    return X, y


def evaluate(model, X, y, name: str):
    proba = model.predict_proba(X)
    pred = model.predict(X)
    print(f"\n--- {name} (n={len(y):,}) ---")
    print(classification_report(y, pred, digits=3, zero_division=0))
    print("confusion matrix (행=실제, 열=예측):")
    print(confusion_matrix(y, pred))
    try:
        if proba.shape[1] == 2:
            auc = roc_auc_score(y, proba[:, 1])
        else:
            auc = roc_auc_score(y, proba, average="macro", multi_class="ovr")
        print(f"AUC: {auc:.3f}")
    except ValueError as e:
        auc = float("nan")
        print(f"AUC 계산 불가: {e}")
    return auc
