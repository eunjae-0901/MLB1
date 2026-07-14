"""
모델 4. LSTM 기반 DNN, '그 외'(label=3) 제거 후 이진분류(0=안다침 / 1=어깨·팔꿈치)
*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 ***
입력: data/processed/{role}_sequence_arrays.npz (경기 순서를 보존한 시계열)

실행: python 04_dnn_lstm_no_other.py --role bullpen
      python 04_dnn_lstm_no_other.py --role starter
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_dnn import (  # noqa: E402
    DEVICE, MODEL_DIR, InjuryLSTM, evaluate, eval_val_auc_only, load_role_arrays, make_datasets,
)


def run(role: str, epochs: int = 100, batch_size: int = 512, lr: float = 1e-3, patience: int = 10):
    data, meta = load_role_arrays(role, exclude_other=True)
    datasets = make_datasets(data, meta)

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)
    test_loader = DataLoader(datasets["test"], batch_size=batch_size)

    model = InjuryLSTM(
        seq_feat_dim=data["X_seq"].shape[2],
        static_feat_dim=data["X_static"].shape[1],
        p_throws_vocab=meta["p_throws_vocab_size"],
        country_vocab=meta["country_vocab_size"],
        head_hidden_sizes=(64,),
        num_classes=2,
    ).to(DEVICE)

    y_train = data["y"][data["split"] == "train"]
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"[모델4] LSTM 이진분류(그 외 제외) - {role.upper()}  device={DEVICE}")
    print(f"train={len(datasets['train']):,} val={len(datasets['val']):,} test={len(datasets['test']):,}")
    print(f"class weights: {dict(zip([0, 1], class_weights.round(2)))}")

    best_val_auc, best_state, patience_left = -1.0, None, patience
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_seq, lengths, x_static, cat_p, cat_c, y in train_loader:
            x_seq, lengths = x_seq.to(DEVICE), lengths.to(DEVICE)
            x_static, cat_p, cat_c, y = (
                x_static.to(DEVICE), cat_p.to(DEVICE), cat_c.to(DEVICE), y.to(DEVICE)
            )
            optimizer.zero_grad()
            logits = model(x_seq, lengths, x_static, cat_p, cat_c)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

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

    model_path = MODEL_DIR / f"04_{role}_dnn_lstm_no_other.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[saved] {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["bullpen", "starter"], required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    run(args.role, args.epochs, args.batch_size, args.lr, args.patience)
