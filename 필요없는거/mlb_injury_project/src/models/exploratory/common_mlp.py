"""
07_dnn_mlp.py가 쓰는 공통 유틸. rolling-window로 평균 낸 표 데이터
(bullpen/starter_window_dataset.parquet - XGBoost 01/02/06번과 완전히 같은 입력)를
LSTM 없이 그냥 MLP(전결합 신경망)로 학습시키기 위한 데이터 로딩/모델 정의.
*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 ***
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_xgb import CATEGORICAL_COLS, DROP_COLS, MODEL_DIR, PROCESSED_DIR  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_role_tabular(role: str, exclude_other: bool = False, binarize: bool = True):
    """window_dataset.parquet을 읽어서 MLP 입력용으로 정리.
    exclude_other/binarize 의미는 common_xgb.load_role과 동일하다."""
    path = PROCESSED_DIR / f"{role}_window_dataset.parquet"
    df = pd.read_parquet(path)

    if exclude_other:
        df = df[df["label"] != 3].copy()
        if binarize:
            df["label"] = (df["label"] > 0).astype(int)

    num_cols = [c for c in df.columns if c not in DROP_COLS + CATEGORICAL_COLS]

    # 범주형은 train 기준으로 vocab을 고정(0=미확인/결측 예약)
    train_df = df[df["split"] == "train"]
    vocabs = {
        c: {v: i + 1 for i, v in enumerate(sorted(train_df[c].dropna().unique()))}
        for c in CATEGORICAL_COLS
    }

    data = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        data[split] = {
            "X_num": sub[num_cols].astype(np.float32).fillna(0.0).to_numpy(),
            "X_cat": np.stack(
                [sub[c].map(vocabs[c]).fillna(0).astype(np.int64).to_numpy() for c in CATEGORICAL_COLS],
                axis=1,
            ),
            "y": sub["label"].astype(np.int64).to_numpy(),
        }

    meta = {
        "num_cols": num_cols,
        "cat_cols": CATEGORICAL_COLS,
        "cat_vocab_sizes": [len(vocabs[c]) + 1 for c in CATEGORICAL_COLS],
        "n_classes": len(set(df["label"].tolist())),
    }
    return data, meta


def standardize(train_arr, *other_arrs):
    """train 통계로만 표준화 (val/test는 train 평균/표준편차를 그대로 적용, leakage 방지)."""
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-6
    out = [(train_arr - mean) / std]
    for arr in other_arrs:
        out.append((arr - mean) / std)
    return out


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


def make_datasets(data, meta):
    X_num_train, X_num_val, X_num_test = standardize(
        data["train"]["X_num"], data["val"]["X_num"], data["test"]["X_num"]
    )
    return {
        "train": TabularDataset(X_num_train, data["train"]["X_cat"], data["train"]["y"]),
        "val": TabularDataset(X_num_val, data["val"]["X_cat"], data["val"]["y"]),
        "test": TabularDataset(X_num_test, data["test"]["X_cat"], data["test"]["y"]),
    }


class TabularMLP(nn.Module):
    """rolling-window 평균 표 데이터 전용 MLP. LSTM 없이 정적 변수만 다룬다는 점이
    InjuryLSTM(common_dnn.py)과 다르다 - 그쪽은 경기 순서(시계열)까지 같이 본다."""

    def __init__(self, num_dim, cat_vocab_sizes, embed_dims=None, hidden_sizes=(128, 64), num_classes=2, dropout=0.3):
        super().__init__()
        embed_dims = embed_dims or [min(8, (v // 2) + 1) for v in cat_vocab_sizes]
        self.embeds = nn.ModuleList([
            nn.Embedding(vocab, dim, padding_idx=0)
            for vocab, dim in zip(cat_vocab_sizes, embed_dims)
        ])
        prev = num_dim + sum(embed_dims)
        layers = []
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
        combined = torch.cat([x_num] + embedded, dim=1)
        return self.head(combined)


def evaluate(model, loader, name):
    model.eval()
    all_y, all_pred, all_proba = [], [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            proba = torch.softmax(model(x_num, x_cat), dim=1).cpu().numpy()
            all_y.append(y.numpy())
            all_pred.append(proba.argmax(axis=1))
            all_proba.append(proba)

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    y_proba = np.concatenate(all_proba)

    print(f"\n--- {name} (n={len(y_true):,}) ---")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))
    print("confusion matrix (행=실제, 열=예측):")
    print(confusion_matrix(y_true, y_pred))
    try:
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")
        print(f"AUC: {auc:.3f}")
    except ValueError as e:
        auc = float("nan")
        print(f"AUC 계산 불가: {e}")
    return auc


def eval_val_auc_only(model, loader) -> float:
    model.eval()
    all_y, all_proba = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
            all_proba.append(torch.softmax(model(x_num, x_cat), dim=1).cpu().numpy())
            all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_proba = np.concatenate(all_proba)
    try:
        if y_proba.shape[1] == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        return 0.0
