"""
모델 7. rolling-window로 평균 낸 표 데이터(bullpen/starter_window_dataset.parquet -
XGBoost 01/02/06번과 완전히 같은 입력)를 LSTM 없이 그냥 MLP(전결합 신경망)로 학습.
*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 ***

모델 3(LSTM)과 이 모델의 차이: 둘 다 "딥러닝"이지만, 모델 3은 경기 순서를 보존한
시계열 입력(.npz)에 LSTM을 쓰고, 이 모델은 평균으로 뭉갠 표 데이터에 MLP만 쓴다.
"LSTM이라서 좋아지는지 / 그냥 딥러닝이라서 좋아지는지"를 나눠서 보기 위한 비교용.

--label_mode 옵션:
  4class          : 0/1/2/3 그대로 4종 분류 (모델 1과 동일 조건)
  binary          : label==3 제거 후 0 vs 1(어깨+팔꿈치) 이진분류 (모델 2와 동일 조건)
  shoulder_elbow  : label==3 제거 후 0/1/2 3종 분류 (모델 6과 동일 조건)

실행: python 07_dnn_mlp.py --role bullpen --label_mode 4class
      python 07_dnn_mlp.py --role starter --label_mode binary
      python 07_dnn_mlp.py --role bullpen --label_mode shoulder_elbow
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from common_mlp import (
    DEVICE, MODEL_DIR, TabularMLP, evaluate, load_role_tabular, make_datasets,
)

LABEL_MODE_ARGS = {
    "4class": dict(exclude_other=False, binarize=True),   # binarize 인자는 exclude_other=False라 무시됨
    "binary": dict(exclude_other=True, binarize=True),
    "shoulder_elbow": dict(exclude_other=True, binarize=False),
}


def run(role: str, label_mode: str, epochs: int = 100, batch_size: int = 512,
        lr: float = 1e-3, patience: int = 10):
    data, meta = load_role_tabular(role, **LABEL_MODE_ARGS[label_mode])
    datasets = make_datasets(data, meta)
    n_classes = meta["n_classes"]

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size)

    model = TabularMLP(
        num_dim=data["train"]["X_num"].shape[1],
        cat_vocab_sizes=meta["cat_vocab_sizes"],
        hidden_sizes=(128, 64),
        num_classes=n_classes,
    ).to(DEVICE)

    y_train = data["train"]["y"]
    classes = np.array(sorted(set(y_train.tolist())))
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"[모델7] MLP({label_mode}) rolling-window 데이터 - {role.upper()}  device={DEVICE}")
    print(f"train={len(datasets['train']):,} val={len(datasets['val']):,} test={len(datasets['test']):,}")
    print(f"class weights: {dict(zip(classes.tolist(), class_weights.round(2)))}")

    best_val_auc, best_state, patience_left = -1.0, None, patience
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_num, x_cat), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        from common_mlp import eval_val_auc_only
        val_auc = eval_val_auc_only(model, val_loader)
        print(f"epoch {epoch:3d}  train_loss={total_loss/len(datasets['train']):.4f}  val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc, best_state, patience_left = val_auc, model.state_dict(), patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping (best val_auc={best_val_auc:.4f})")
                break

    model.load_state_dict(best_state)
    evaluate(model, val_loader, "Validation (best epoch)")
    evaluate(model, test_loader, "Test")

    model_path = MODEL_DIR / f"07_{role}_dnn_mlp_{label_mode}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[saved] {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["bullpen", "starter"], required=True)
    parser.add_argument("--label_mode", choices=list(LABEL_MODE_ARGS), default="4class")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    run(args.role, args.label_mode, args.epochs, args.batch_size, args.lr, args.patience)
