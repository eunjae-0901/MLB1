"""
DNN 계열(03, 04, 05) 스크립트가 공유하는 데이터 로딩/모델/평가 유틸.
*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 ***
필요 패키지: torch, numpy, pandas, scikit-learn
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, X_seq, mask, X_static, cat_p_throws, cat_country, y):
        self.X_seq = torch.as_tensor(X_seq, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.X_static = torch.as_tensor(X_static, dtype=torch.float32)
        self.cat_p_throws = torch.as_tensor(cat_p_throws, dtype=torch.long)
        self.cat_country = torch.as_tensor(cat_country, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.lengths = self.mask.sum(dim=1).clamp(min=1).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_seq[idx], self.lengths[idx], self.X_static[idx],
            self.cat_p_throws[idx], self.cat_country[idx], self.y[idx],
        )


class InjuryLSTM(nn.Module):
    """LSTM으로 경기 순서를 인코딩 + 정적변수를 붙여 분류하는 기본 구조.
    head(마지막 MLP)는 hidden_sizes 리스트로 층 구성을 자유롭게 넣을 수 있다
    (05번 베이지안 최적화에서 이 리스트를 탐색 대상으로 씀)."""

    def __init__(self, seq_feat_dim, static_feat_dim, p_throws_vocab, country_vocab,
                 lstm_hidden=64, head_hidden_sizes=(64,), num_classes=2, dropout=0.3):
        super().__init__()
        self.embed_p_throws = nn.Embedding(p_throws_vocab, 2, padding_idx=0)
        self.embed_country = nn.Embedding(country_vocab, 4, padding_idx=0)
        self.lstm = nn.LSTM(seq_feat_dim, lstm_hidden, num_layers=1, batch_first=True)

        combined_dim = lstm_hidden + 2 + 4 + static_feat_dim
        layers = []
        prev = combined_dim
        for h in head_hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x_seq, lengths, x_static, cat_p_throws, cat_country):
        packed = pack_padded_sequence(
            x_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        seq_repr = h_n[-1]
        combined = torch.cat([
            seq_repr,
            self.embed_p_throws(cat_p_throws),
            self.embed_country(cat_country),
            x_static,
        ], dim=1)
        return self.head(combined)


def load_role_arrays(role: str, exclude_other: bool = False):
    """.npz + meta.json을 읽어서 dict로 반환. exclude_other=True면 label==3 행을
    제거하고 이진(0/1)으로 재라벨링한다."""
    npz = np.load(PROCESSED_DIR / f"{role}_sequence_arrays.npz", allow_pickle=True)
    meta = json.loads((PROCESSED_DIR / f"{role}_sequence_meta.json").read_text(encoding="utf-8"))

    data = {k: npz[k] for k in npz.files}
    if exclude_other:
        keep = data["y"] != 3
        for k in ("X_seq", "mask", "X_static", "cat_p_throws", "cat_country", "y", "split"):
            data[k] = data[k][keep]
        data["y"] = (data["y"] > 0).astype(np.int64)
        meta["n_classes"] = 2
    else:
        meta["n_classes"] = 4
    return data, meta


def standardize(train_arr, *other_arrs):
    """train 기준 평균/표준편차로 표준화 (val/test는 train 통계 그대로, leakage 방지)."""
    mean = train_arr.mean(axis=tuple(range(train_arr.ndim - 1)), keepdims=True)
    std = train_arr.std(axis=tuple(range(train_arr.ndim - 1)), keepdims=True) + 1e-6
    out = [(train_arr - mean) / std]
    for arr in other_arrs:
        out.append((arr - mean) / std)
    return out


def make_datasets(data, meta):
    split = data["split"]
    train_idx, val_idx, test_idx = split == "train", split == "val", split == "test"

    X_seq_train, X_seq_val, X_seq_test = standardize(
        data["X_seq"][train_idx], data["X_seq"][val_idx], data["X_seq"][test_idx]
    )
    X_static_train, X_static_val, X_static_test = standardize(
        data["X_static"][train_idx], data["X_static"][val_idx], data["X_static"][test_idx]
    )

    def ds(seq, static, idx):
        return SequenceDataset(
            seq, data["mask"][idx], static,
            data["cat_p_throws"][idx], data["cat_country"][idx], data["y"][idx],
        )

    return {
        "train": ds(X_seq_train, X_static_train, train_idx),
        "val": ds(X_seq_val, X_static_val, val_idx),
        "test": ds(X_seq_test, X_static_test, test_idx),
    }


def evaluate(model, loader, name):
    model.eval()
    all_y, all_pred, all_proba = [], [], []
    with torch.no_grad():
        for x_seq, lengths, x_static, cat_p, cat_c, y in loader:
            x_seq, lengths = x_seq.to(DEVICE), lengths.to(DEVICE)
            x_static, cat_p, cat_c = x_static.to(DEVICE), cat_p.to(DEVICE), cat_c.to(DEVICE)
            logits = model(x_seq, lengths, x_static, cat_p, cat_c)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
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
    """학습 중 매 epoch마다 빠르게 val AUC만 계산 (베이지안 최적화 objective용)."""
    model.eval()
    all_y, all_proba = [], []
    with torch.no_grad():
        for x_seq, lengths, x_static, cat_p, cat_c, y in loader:
            x_seq, lengths = x_seq.to(DEVICE), lengths.to(DEVICE)
            x_static, cat_p, cat_c = x_static.to(DEVICE), cat_p.to(DEVICE), cat_c.to(DEVICE)
            logits = model(x_seq, lengths, x_static, cat_p, cat_c)
            all_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_y.append(y.numpy())
    y_true = np.concatenate(all_y)
    y_proba = np.concatenate(all_proba)
    try:
        if y_proba.shape[1] == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")
    except ValueError:
        return 0.0
