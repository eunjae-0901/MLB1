"""
모델 2 (공식 파이프라인 1단계). DNN(MLP) + Bayesian Optimization, rolling-window
데이터(bullpen/starter_window_dataset.parquet - 01_xgboost_bayesian.py와 완전히
같은 입력/전처리). '그 외'(label=3) 행은 항상 제외하고, --label_mode로 두 가지
분류 방식을 둘 다 지원한다(01번과 동일).
  3class : 0(안다침)/1(어깨)/2(팔꿈치) 3종 분류
  binary : 1과 2를 합쳐서 0(안다침) vs 1(어깨 또는 팔꿈치) 이진분류
*** GPU/torch가 설치된 별도 가상환경에서 실행할 것 ***

01번과 다른 점은 딱 하나, 모델을 XGBoost 대신 딥러닝(MLP)으로 바꾼 것뿐이다.
데이터/전처리/라벨 기준을 동일하게 맞춰야 두 모델 성능을 공정하게 비교할 수 있다.
- LSTM이 아니라 MLP를 쓰는 이유: rolling-window 데이터는 이미 14일/3경기를 평균으로
  뭉갠 표 데이터라 경기 순서(시계열) 정보 자체가 없다. 그래서 LSTM을 쓸 이유가 없고,
  경기 순서를 보존한 시계열 입력(.npz)에 LSTM을 쓰는 실험은 exploratory/에 있는
  05_dnn_lstm_bayesian.py 쪽에서 별도로 진행한다.

은닉층 구조 결정 로직(determine_hidden_sizes)과 베이지안 탐색 방식은 exploratory의
05_dnn_lstm_bayesian.py에서 그대로 가져왔다(첫 은닉층 노드 수/은닉층 수 두 값만
정하면 나머지 층 크기가 등차수열로 자동 결정되는 방식).

탐색 대상 하이퍼파라미터 5개:
  learning_rate      (1e-4, 1e-2, log scale)
  hidden_layer_init   (1, 5)     : 은닉층 수(h)
  hidden_node_init    (8, 128)   : 첫 은닉층 노드 수(n_1)
  batch_size          (128, 1024)
  dropout             (0.1, 0.5)

입력변수 정리: 01번과 동일하게 상관계수 0.9 초과 컬럼을 먼저 제거하고, DNN 쪽은
추가로 PCA(설명 분산 95%)까지 적용해서 차원을 한 번 더 줄인다 - 표본(특히 어깨/팔꿈치
양성 표본) 대비 입력변수가 너무 많으면 딥러닝이 과적합하기 쉽다는 판단.

중단 후 이어하기(resume): GPU 세션이 길게 도는 도중 끊길 수 있어서, 아래 두 단계 모두
중간 저장을 해둔다. 다시 실행하면(같은 --role --label_mode로) 자동으로 이어서 진행하고,
처음부터 다시 돌리지 않는다.
  1) 베이지안 탐색 단계: trial이 끝날 때마다 바로
     models/02_{role}_dnn_bayesian_{label_mode}_trials.json에 이어붙여 저장한다.
     재실행 시 이 파일에 있는 trial들을 optimizer에 등록해두고, 목표 횟수에서 이미
     한 만큼을 뺀 나머지만 더 탐색한다.
  2) 최적 하이퍼파라미터로 본학습하는 단계: 매 epoch마다
     models/02_{role}_dnn_bayesian_{label_mode}_checkpoint.pt에 모델/옵티마이저
     상태를 저장한다. 재실행 시 이 체크포인트를 불러와 그 다음 epoch부터 이어서
     학습한다(단, 하이퍼파라미터 구성이 이전과 다르면 무시하고 처음부터 시작).
     본학습이 정상적으로 끝나면 이 체크포인트 파일은 필요 없어져서 자동 삭제된다.

실행: python 02_dnn_bayesian.py --role bullpen --label_mode 3class --n_iter 25
      python 02_dnn_bayesian.py --role bullpen --label_mode binary --n_iter 25
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
from data_utils import MODEL_DIR  # noqa: E402
from dnn_bayesian_utils import (  # noqa: E402
    DEVICE, TabularMLP, determine_hidden_sizes, eval_macro_auc, evaluate, prepare_data,
)


def load_prior_trials(role: str, label_mode: str) -> list[dict]:
    """이전에 중단된 실행이 남긴 trial 기록이 있으면 불러온다(베이지안 탐색 이어하기용)."""
    trials_path = MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}_trials.json"
    if trials_path.exists():
        return json.loads(trials_path.read_text(encoding="utf-8"))
    return []


def append_trial(role: str, label_mode: str, params: dict, target: float):
    """trial 하나가 끝날 때마다 즉시 파일에 이어붙여 저장 -> 중간에 끊겨도 여기까지는 보존됨."""
    trials_path = MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}_trials.json"
    existing = load_prior_trials(role, label_mode)
    existing.append({"target": target, "params": params})
    trials_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def make_objective(role, label_mode, datasets, meta, class_weights, quick_epochs, quick_patience):
    def objective(learning_rate, hidden_layer_init, hidden_node_init, batch_size, dropout):
        batch_size = int(batch_size)
        hidden_sizes = determine_hidden_sizes(meta["num_dim"], hidden_layer_init, hidden_node_init)
        lr = 10 ** learning_rate

        train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=512)

        model = TabularMLP(
            num_dim=meta["num_dim"], p_throws_vocab=meta["p_throws_vocab_size"],
            country_vocab=meta["country_vocab_size"], hidden_sizes=hidden_sizes,
            num_classes=meta["n_classes"], dropout=dropout,
        ).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        best_val_auc, patience_left = -1.0, quick_patience
        for _ in range(quick_epochs):
            model.train()
            for x_num, x_cat, y in train_loader:
                x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x_num, x_cat), y)
                loss.backward()
                optimizer.step()

            val_auc = eval_macro_auc(model, val_loader)
            if val_auc > best_val_auc:
                best_val_auc, patience_left = val_auc, quick_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        print(f"  [trial] h={hidden_layer_init:.2f} n1={hidden_node_init:.2f} -> "
              f"층구성={hidden_sizes}  lr={lr:.2e}  batch={batch_size}  dropout={dropout:.2f}  "
              f"val_auc={best_val_auc:.4f}")

        append_trial(role, label_mode, {
            "learning_rate": learning_rate, "hidden_layer_init": hidden_layer_init,
            "hidden_node_init": hidden_node_init, "batch_size": batch_size, "dropout": dropout,
        }, best_val_auc)
        return best_val_auc

    return objective


PBOUNDS = {
    "learning_rate": (-4.0, -2.0),
    "hidden_layer_init": (1, 5),
    "hidden_node_init": (8, 128),
    "batch_size": (128, 1024),
    "dropout": (0.1, 0.5),
}


def run(role: str, label_mode: str, n_iter: int, init_points: int, quick_epochs: int, quick_patience: int,
        final_epochs: int, final_patience: int):
    print(f"\n{'=' * 70}\n[모델2:{label_mode}] DNN(MLP) + Bayesian Optimization - "
          f"{role.upper()}  device={DEVICE}\n{'=' * 70}")
    datasets, meta, y_train, _splits = prepare_data(role, label_mode)
    print(f"train={len(datasets['train']):,} val={len(datasets['val']):,} test={len(datasets['test']):,}")
    print(f"PCA 이후 입력 차원: {meta['num_dim']}  클래스 수: {meta['n_classes']}")

    class_weights = compute_class_weight("balanced", classes=np.arange(meta["n_classes"]), y=y_train)

    objective = make_objective(role, label_mode, datasets, meta, class_weights, quick_epochs, quick_patience)
    optimizer = BayesianOptimization(f=objective, pbounds=PBOUNDS, random_state=42, verbose=2)

    # 이전에 중단된 실행이 남긴 trial이 있으면 이어서 진행 (처음부터 다시 안 돌려도 됨)
    prior_trials = load_prior_trials(role, label_mode)
    for t in prior_trials:
        try:
            optimizer.register(params=t["params"], target=t["target"])
        except KeyError:
            pass  # pbounds가 바뀐 등으로 이전 기록과 파라미터 이름이 안 맞으면 그 trial은 건너뜀
    total_wanted = init_points + n_iter
    already_done = len(optimizer.res)
    remaining = max(0, total_wanted - already_done)
    if prior_trials:
        print(f"[이어하기] 이전 trial {len(prior_trials)}개 발견 -> 등록하고 나머지 {remaining}개만 더 탐색")

    if remaining > 0:
        optimizer.maximize(init_points=0 if prior_trials else init_points,
                            n_iter=remaining if prior_trials else n_iter)

    best = optimizer.max
    print(f"\n베이지안 최적화 최종 결과: {best}")

    p = best["params"]
    hidden_sizes = determine_hidden_sizes(meta["num_dim"], p["hidden_layer_init"], p["hidden_node_init"])
    batch_size = int(p["batch_size"])
    lr = 10 ** p["learning_rate"]
    print(f"\n최종 모델 은닉층 구성: {hidden_sizes}  lr={lr:.2e}  batch={batch_size}  dropout={p['dropout']:.2f}")

    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=512)
    test_loader = DataLoader(datasets["test"], batch_size=512)

    model = TabularMLP(
        num_dim=meta["num_dim"], p_throws_vocab=meta["p_throws_vocab_size"],
        country_vocab=meta["country_vocab_size"], hidden_sizes=hidden_sizes,
        num_classes=meta["n_classes"], dropout=p["dropout"],
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
    final_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 본학습 도중 끊겨도 이어할 수 있도록 매 epoch 체크포인트를 저장한다. 재실행 시
    # 같은 하이퍼파라미터(같은 은닉층 구성/배치사이즈)로 이어받은 체크포인트가 있으면
    # 그 epoch부터 다시 시작한다 - 베이지안 탐색 결과가 달라지면(재탐색 등) 구조가 안
    # 맞을 수 있으니 그럴 땐 체크포인트를 무시하고 처음부터 시작한다.
    ckpt_path = MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}_checkpoint.pt"
    start_epoch = 1
    best_val_auc, best_state, patience_left = -1.0, None, final_patience
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        if ckpt["hidden_sizes"] == hidden_sizes and ckpt["batch_size"] == batch_size:
            model.load_state_dict(ckpt["model_state"])
            final_optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_auc = ckpt["best_val_auc"]
            best_state = ckpt["best_state"]
            patience_left = ckpt["patience_left"]
            print(f"[이어하기] 체크포인트 발견 -> epoch {start_epoch}부터 이어서 본학습 진행"
                  f" (그때까지 best_val_auc={best_val_auc:.4f})")
        else:
            print("[알림] 체크포인트가 있지만 하이퍼파라미터 구성이 달라서 무시하고 처음부터 시작")

    for epoch in range(start_epoch, final_epochs + 1):
        model.train()
        total_loss = 0.0
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            final_optimizer.zero_grad()
            loss = criterion(model(x_num, x_cat), y)
            loss.backward()
            final_optimizer.step()
            total_loss += loss.item() * len(y)

        val_auc = eval_macro_auc(model, val_loader)
        print(f"epoch {epoch:3d}  train_loss={total_loss/len(datasets['train']):.4f}  val_auc={val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc, best_state, patience_left = val_auc, model.state_dict(), final_patience
        else:
            patience_left -= 1

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": final_optimizer.state_dict(),
            "best_val_auc": best_val_auc, "best_state": best_state,
            "patience_left": patience_left, "hidden_sizes": hidden_sizes, "batch_size": batch_size,
        }, ckpt_path)

        if patience_left <= 0:
            print(f"early stopping (best val_auc={best_val_auc:.4f})")
            break

    model.load_state_dict(best_state)
    val_auc = evaluate(model, val_loader, "Validation (최적 하이퍼파라미터)")
    test_auc = evaluate(model, test_loader, "Test (최적 하이퍼파라미터)")

    model_path = MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[saved] {model_path}")

    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"[정리] 본학습 완료 -> 체크포인트 파일 삭제({ckpt_path.name})")

    summary_path = MODEL_DIR / f"02_{role}_dnn_bayesian_{label_mode}_summary.json"
    summary_path.write_text(
        json.dumps({
            "role": role, "label_mode": label_mode,
            "kept_num_cols": meta["kept_num_cols"], "pca_dim": meta["num_dim"],
            "n_classes": meta["n_classes"],
            "best_params": p, "hidden_sizes": hidden_sizes,
            "val_auc": val_auc, "test_auc": test_auc,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["bullpen", "starter"], required=True)
    parser.add_argument("--label_mode", choices=["3class", "binary", "both"], default="both")
    parser.add_argument("--n_iter", type=int, default=25, help="베이지안 탐색 반복 횟수")
    parser.add_argument("--init_points", type=int, default=8, help="초기 무작위 탐색 횟수")
    parser.add_argument("--quick_epochs", type=int, default=25, help="탐색 단계 trial당 최대 epoch")
    parser.add_argument("--quick_patience", type=int, default=5)
    parser.add_argument("--final_epochs", type=int, default=100, help="최적 조합 확정 후 본학습 epoch")
    parser.add_argument("--final_patience", type=int, default=10)
    args = parser.parse_args()

    label_modes = ["3class", "binary"] if args.label_mode == "both" else [args.label_mode]
    for label_mode in label_modes:
        run(args.role, label_mode, args.n_iter, args.init_points, args.quick_epochs, args.quick_patience,
            args.final_epochs, args.final_patience)
