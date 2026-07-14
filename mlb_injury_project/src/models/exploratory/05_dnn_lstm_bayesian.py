"""
모델 5. LSTM 기반 DNN + 베이지안 최적화로 은닉층 구조/학습률/배치사이즈 탐색.
'그 외'(label=3) 제거 후 이진분류(0=안다침 / 1=어깨·팔꿈치) - 모델 4와 동일한 문제 설정.

*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 ***
필요 패키지: torch, numpy, pandas, scikit-learn, bayesian-optimization (pip install bayesian-optimization)

은닉노드 결정 로직은 DNN_bayesian_1.ipynb(선행 참고자료, WLTP 데이터용 Keras 코드)의
로직을 그대로 포팅했다 - 첫 은닉층 노드 수(n_1)와 은닉층 수(h)만 정하면, 나머지 층의
노드 수는 "등차수열로 균등하게 분산"시키는 규칙으로 자동 결정된다.
  - n_1 <= input_dim: 층이 늘어날수록 노드 수가 등차수열로 감소
  - n_1 >  input_dim: 노드 수가 등차수열로 증가했다가 다시 감소 (중간이 볼록한 모양)
참고자료는 TensorFlow/Keras였는데, 이 스크립트는 PyTorch로 다시 구현했다.

실행: python 05_dnn_lstm_bayesian.py --role bullpen --n_iter 30
      python 05_dnn_lstm_bayesian.py --role starter --n_iter 30
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_dnn import (  # noqa: E402
    DEVICE, MODEL_DIR, InjuryLSTM, evaluate, eval_val_auc_only, load_role_arrays, make_datasets,
)

LSTM_HIDDEN = 64
EMBED_DIM = 2 + 4  # p_throws + country


def determine_hidden_sizes(input_dim: int, hidden_layer_init: float, hidden_node_init: float) -> list[int]:
    """DNN_bayesian_1.ipynb의 은닉노드 결정 로직을 그대로 포팅 (Table 1/2 로직)."""
    h = max(int(round(hidden_layer_init)), 1)
    n1 = max(int(round(hidden_node_init)), 1)

    if n1 <= input_dim:
        # Case: 노드 수가 감소하는 등차수열
        if n1 % h == 0:
            n_hidden_layer = h
            step = n1 // n_hidden_layer
        else:
            n_hidden_layer = h + 1
            step = n1 // n_hidden_layer
        sizes = [max(n1 - step * i, 1) for i in range(n_hidden_layer)]
    else:
        # Case: 노드 수가 증가했다가 감소하는 등차수열 (중간이 볼록)
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


def make_objective(datasets, data, meta, seq_feat_dim, static_feat_dim, class_weights, quick_epochs, quick_patience):
    combined_dim = LSTM_HIDDEN + EMBED_DIM + static_feat_dim

    def objective(learning_rate, hidden_layer_init, hidden_node_init, batch_size, dropout):
        batch_size = int(batch_size)
        hidden_sizes = determine_hidden_sizes(combined_dim, hidden_layer_init, hidden_node_init)
        lr = 10 ** learning_rate

        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=512)

        model = InjuryLSTM(
            seq_feat_dim=seq_feat_dim, static_feat_dim=static_feat_dim,
            p_throws_vocab=meta["p_throws_vocab_size"], country_vocab=meta["country_vocab_size"],
            lstm_hidden=LSTM_HIDDEN, head_hidden_sizes=hidden_sizes, num_classes=2, dropout=dropout,
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        best_val_auc, patience_left = -1.0, quick_patience
        for _ in range(quick_epochs):
            model.train()
            for x_seq, lengths, x_static, cat_p, cat_c, y in train_loader:
                x_seq, lengths = x_seq.to(DEVICE), lengths.to(DEVICE)
                x_static, cat_p, cat_c, y = (
                    x_static.to(DEVICE), cat_p.to(DEVICE), cat_c.to(DEVICE), y.to(DEVICE)
                )
                optimizer.zero_grad()
                loss = criterion(model(x_seq, lengths, x_static, cat_p, cat_c), y)
                loss.backward()
                optimizer.step()

            val_auc = eval_val_auc_only(model, val_loader)
            if val_auc > best_val_auc:
                best_val_auc, patience_left = val_auc, quick_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        print(f"  [trial] h={hidden_layer_init:.2f} n1={hidden_node_init:.2f} -> "
              f"층구성={hidden_sizes}  lr={lr:.2e}  batch={batch_size}  dropout={dropout:.2f}  "
              f"val_auc={best_val_auc:.4f}")
        return best_val_auc

    return objective


def run(role: str, n_iter: int, quick_epochs: int, quick_patience: int,
        final_epochs: int, final_patience: int):
    data, meta = load_role_arrays(role, exclude_other=True)
    datasets = make_datasets(data, meta)
    seq_feat_dim = data["X_seq"].shape[2]
    static_feat_dim = data["X_static"].shape[1]

    y_train = data["y"][data["split"] == "train"]
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)

    print(f"[모델5] LSTM + 베이지안 최적화(그 외 제외) - {role.upper()}  device={DEVICE}")
    print(f"train={len(datasets['train']):,} val={len(datasets['val']):,} test={len(datasets['test']):,}")

    objective = make_objective(
        datasets, data, meta, seq_feat_dim, static_feat_dim, class_weights,
        quick_epochs, quick_patience,
    )

    pbounds = {
        "learning_rate": (-4.0, -2.0),      # 10^-4 ~ 10^-2
        "hidden_layer_init": (1, 5),        # 은닉층 수(h)
        "hidden_node_init": (8, 128),       # 첫 은닉층 노드 수(n_1)
        "batch_size": (128, 1024),
        "dropout": (0.1, 0.5),
    }

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42, verbose=2)
    optimizer.maximize(init_points=5, n_iter=n_iter)

    best = optimizer.max
    print(f"\n베이지안 최적화 최종 결과: {best}")

    result_path = MODEL_DIR / f"05_{role}_bayesian_trials.json"
    result_path.write_text(
        json.dumps([{"target": r["target"], "params": r["params"]} for r in optimizer.res],
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {result_path}")

    # 최적 하이퍼파라미터로 본 학습(더 긴 epoch)
    p = best["params"]
    combined_dim = LSTM_HIDDEN + EMBED_DIM + static_feat_dim
    hidden_sizes = determine_hidden_sizes(combined_dim, p["hidden_layer_init"], p["hidden_node_init"])
    batch_size = int(p["batch_size"])
    lr = 10 ** p["learning_rate"]
    print(f"\n최종 모델 은닉층 구성: {hidden_sizes}  lr={lr:.2e}  batch={batch_size}  dropout={p['dropout']:.2f}")

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=512)
    test_loader = DataLoader(datasets["test"], batch_size=512)

    model = InjuryLSTM(
        seq_feat_dim=seq_feat_dim, static_feat_dim=static_feat_dim,
        p_throws_vocab=meta["p_throws_vocab_size"], country_vocab=meta["country_vocab_size"],
        lstm_hidden=LSTM_HIDDEN, head_hidden_sizes=hidden_sizes, num_classes=2, dropout=p["dropout"],
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    final_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc, best_state, patience_left = -1.0, None, final_patience
    for epoch in range(1, final_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_seq, lengths, x_static, cat_p, cat_c, y in train_loader:
            x_seq, lengths = x_seq.to(DEVICE), lengths.to(DEVICE)
            x_static, cat_p, cat_c, y = (
                x_static.to(DEVICE), cat_p.to(DEVICE), cat_c.to(DEVICE), y.to(DEVICE)
            )
            final_optimizer.zero_grad()
            loss = criterion(model(x_seq, lengths, x_static, cat_p, cat_c), y)
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item() * len(y)

        val_auc = eval_val_auc_only(model, val_loader)
        print(f"epoch {epoch:3d}  train_loss={total_loss/len(datasets['train']):.4f}  val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc, best_state, patience_left = val_auc, model.state_dict(), final_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping (best val_auc={best_val_auc:.4f})")
                break

    model.load_state_dict(best_state)
    evaluate(model, val_loader, "Validation (best epoch)")
    evaluate(model, test_loader, "Test")

    model_path = MODEL_DIR / f"05_{role}_dnn_lstm_bayesian.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[saved] {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["bullpen", "starter"], required=True)
    parser.add_argument("--n_iter", type=int, default=30, help="베이지안 탐색 반복 횟수")
    parser.add_argument("--quick_epochs", type=int, default=25, help="탐색 단계에서 시도(trial)당 최대 epoch")
    parser.add_argument("--quick_patience", type=int, default=5)
    parser.add_argument("--final_epochs", type=int, default=100, help="최적 조합 확정 후 본학습 epoch")
    parser.add_argument("--final_patience", type=int, default=10)
    args = parser.parse_args()
    run(args.role, args.n_iter, args.quick_epochs, args.quick_patience,
        args.final_epochs, args.final_patience)
