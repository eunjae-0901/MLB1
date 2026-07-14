"""
02_dnn_bayesian.py와 export_dnn_bayesian_results.py가 공유하는 모델/데이터 유틸.
02_dnn_bayesian.py는 파일명이 숫자로 시작해서 `import 02_dnn_bayesian`이 안 되기
때문에(파이썬 모듈명 규칙 위반), 재사용할 부분을 이 파일로 따로 뺐다.
*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 ***
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import MODEL_DIR, load_role, numeric_feature_cols  # noqa: E402
from feature_selection import apply_pca, select_uncorrelated_features  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORR_THRESHOLD = 0.9
PCA_VARIANCE = 0.95
EMBED_DIM_P_THROWS = 2
EMBED_DIM_COUNTRY = 4


def determine_hidden_sizes(input_dim: int, hidden_layer_init: float, hidden_node_init: float) -> list[int]:
    """exploratory/05_dnn_lstm_bayesian.py의 은닉노드 결정 로직을 그대로 포팅."""
    h = max(int(round(hidden_layer_init)), 1)
    n1 = max(int(round(hidden_node_init)), 1)

    if n1 <= input_dim:
        if n1 % h == 0:
            n_hidden_layer = h
            step = n1 // n_hidden_layer
        else:
            n_hidden_layer = h + 1
            step = n1 // n_hidden_layer
        sizes = [max(n1 - step * i, 1) for i in range(n_hidden_layer)]
    else:
        n_hidden_layer = h
        if n_hidden_layer % 2 == 0:
            increase_layers = max(n_hidden_layer // 2 - 1, 1)
        else:
            increase_layers = max(n_hidden_layer // 2, 1)
        decrease_layers = max(n_hidden_layer - increase_layers, 1)

        step_increase = n1 - input_dim
        max_node = n1 + (increase_layers - 1) * step_increase
        step_decrease = max(max_node // (decrease_layers + 1), 1)

        sizes = [max(n1 + step_increase * i, 1) for i in range(increase_layers)]
        sizes += [max(max_node - step_decrease * (i + 1), 1) for i in range(decrease_layers)]

    return sizes


class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.as_tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.as_tensor(X_cat, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


class TabularMLP(nn.Module):
    def __init__(self, num_dim, p_throws_vocab, country_vocab, hidden_sizes, num_classes=3, dropout=0.3):
        super().__init__()
        self.embed_p_throws = nn.Embedding(p_throws_vocab, EMBED_DIM_P_THROWS, padding_idx=0)
        self.embed_country = nn.Embedding(country_vocab, EMBED_DIM_COUNTRY, padding_idx=0)

        prev = num_dim + EMBED_DIM_P_THROWS + EMBED_DIM_COUNTRY
        layers = []
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        combined = torch.cat([
            x_num, self.embed_p_throws(x_cat[:, 0]), self.embed_country(x_cat[:, 1]),
        ], dim=1)
        return self.head(combined)


def prepare_data(role: str, label_mode: str = "3class"):
    """01_xgboost_bayesian.py와 동일한 상관관계 기반 feature selection을 적용한 뒤,
    train 기준으로 표준화하고 PCA까지 적용한 numpy 배열을 반환한다.
    label_mode="3class"면 0/1/2, "binary"면 1·2를 합친 0/1로 라벨링한다."""
    splits = load_role(role, exclude_other=True, binarize=(label_mode == "binary"))
    all_num_cols = numeric_feature_cols(splits["train"])
    kept_num_cols = select_uncorrelated_features(splits["train"], all_num_cols, threshold=CORR_THRESHOLD)

    p_throws_vocab = {v: i + 1 for i, v in enumerate(sorted(splits["train"]["p_throws"].dropna().unique()))}
    country_vocab = {v: i + 1 for i, v in enumerate(sorted(splits["train"]["birth_country"].dropna().unique()))}

    raw = {}
    for s, df in splits.items():
        raw[s] = {
            "X_num": df[kept_num_cols].astype(np.float32).fillna(0.0).to_numpy(),
            "X_cat": np.stack([
                df["p_throws"].map(p_throws_vocab).fillna(0).astype(np.int64).to_numpy(),
                df["birth_country"].map(country_vocab).fillna(0).astype(np.int64).to_numpy(),
            ], axis=1),
            "y": df["label"].astype(np.int64).to_numpy(),
        }

    mean = raw["train"]["X_num"].mean(axis=0, keepdims=True)
    std = raw["train"]["X_num"].std(axis=0, keepdims=True) + 1e-6
    for s in raw:
        raw[s]["X_num"] = (raw[s]["X_num"] - mean) / std

    pca, (train_pca, val_pca, test_pca) = apply_pca(
        raw["train"]["X_num"], raw["val"]["X_num"], raw["test"]["X_num"], n_components=PCA_VARIANCE
    )
    raw["train"]["X_num"], raw["val"]["X_num"], raw["test"]["X_num"] = train_pca, val_pca, test_pca

    datasets = {
        s: TabularDataset(raw[s]["X_num"], raw[s]["X_cat"], raw[s]["y"]) for s in raw
    }
    meta = {
        "num_dim": train_pca.shape[1],
        "p_throws_vocab_size": len(p_throws_vocab) + 1,
        "country_vocab_size": len(country_vocab) + 1,
        "kept_num_cols": kept_num_cols,
        "n_classes": 2 if label_mode == "binary" else 3,
    }
    return datasets, meta, raw["train"]["y"], splits


def _auc_any(y_true, y_proba) -> float:
    """클래스가 2개면 이진 AUC, 3개면 macro AUC(OvR)로 자동 분기."""
    if y_proba.shape[1] == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")


def eval_macro_auc(model, loader) -> float:
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
        return _auc_any(y_true, y_proba)
    except ValueError:
        return 0.0


def evaluate(model, loader, name: str) -> float:
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
        auc = _auc_any(y_true, y_proba)
        print(f"AUC: {auc:.3f}")
    except ValueError as e:
        auc = float("nan")
        print(f"AUC 계산 불가: {e}")
    return auc
