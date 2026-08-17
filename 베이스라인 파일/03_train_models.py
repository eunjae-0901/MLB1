"""Train paper-style player-level TJS baselines and save reproducible results."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "results"
CHECKPOINTS = OUT / "checkpoints"
SEEDS = list(range(100, 1001, 100))


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logistic", "lstm", "vit"], required=True)
    p.add_argument("--delta-mode", choices=["official_full_period", "paper_first_season"], default="official_full_period")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, help="Override paper-code maximum epochs")
    return p.parse_args()


def split_scale(X, y, seed):
    indices = np.arange(len(y))
    tr, temp = train_test_split(indices, test_size=.4, random_state=seed, stratify=y)
    va, te = train_test_split(temp, test_size=.5, random_state=seed, stratify=y[temp])
    scaler = MinMaxScaler().fit(X[tr].reshape(-1, X.shape[-1]))
    transform = lambda z: scaler.transform(X[z].reshape(-1, X.shape[-1])).reshape(len(z), X.shape[1], X.shape[2]).astype("float32")
    return tr, va, te, transform(tr), transform(va), transform(te)


def metric_row(y, probability, model, seed, split):
    pred = (probability >= .5).astype(int)
    return {"model": model, "seed": seed, "split": split, "threshold": .5, "n": len(y), "positive_rate": float(np.mean(y)), "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0), "roc_auc": roc_auc_score(y, probability), "pr_auc": average_precision_score(y, probability)}


def train_logistic(X, y, seeds):
    rows, predictions = [], []
    for seed in seeds:
        tr, va, te, xtr, xva, xte = split_scale(X, y, seed)
        model = LogisticRegression(max_iter=3000, class_weight={0: 1, 1: 5}, random_state=seed)
        model.fit(xtr.reshape(len(tr), -1), y[tr])
        for name, idx, xx in (("validation", va, xva), ("test", te, xte)):
            prob = model.predict_proba(xx.reshape(len(idx), -1))[:, 1]
            rows.append(metric_row(y[idx], prob, "logistic", seed, name))
            predictions.extend({"model": "logistic", "seed": seed, "split": name, "row_index": int(i), "y_true": int(y[i]), "probability": float(p)} for i, p in zip(idx, prob))
    return rows, predictions


def train_deep(X, y, kind, seeds, epoch_override):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    timm = None
    if kind == "vit":
        try:
            import timm as _timm
            timm = _timm
        except ImportError as exc:
            raise SystemExit("ViT requires: pip install timm") from exc

    if not torch.cuda.is_available():
        raise SystemExit("LSTM/ViT reproduction requires a CUDA GPU.")
    device = torch.device("cuda")

    class LSTM(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.rnn = nn.LSTM(n_features, 512, num_layers=4, dropout=.2, batch_first=True, bidirectional=True)
            self.norm = nn.LayerNorm(1024)
            self.fc = nn.Linear(1024, 1)
        def forward(self, x):
            return self.fc(self.norm(self.rnn(x)[0]).mean(dim=1))

    class ViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)
        def forward(self, x):
            x = F.pad(x, (0, 224 - x.shape[2]), value=-5).unsqueeze(1).expand(-1, 3, -1, -1)
            return self.backbone(x)

    config = {
        "lstm": {"batch": 128, "epochs": 500, "lr": 8e-6, "patience": 50, "weight_decay": 1e-5, "t0": 25, "warmup": 10, "gamma": 1.0},
        "vit": {"batch": 16, "epochs": 200, "lr": 4e-6, "patience": 20, "weight_decay": 3e-5, "t0": 20, "warmup": 5, "gamma": .8},
    }[kind]
    if epoch_override:
        config["epochs"] = epoch_override
    rows, predictions = [], []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        tr, va, te, xtr, xva, xte = split_scale(X, y, seed)
        loaders = {}
        for name, xx, idx, shuffle in (("train", xtr, tr, kind != "vit"), ("validation", xva, va, False), ("test", xte, te, False)):
            ds = TensorDataset(torch.from_numpy(xx), torch.from_numpy(y[idx].astype("float32")).unsqueeze(1))
            loaders[name] = DataLoader(ds, batch_size=config["batch"], shuffle=shuffle)
        model = (LSTM(X.shape[-1]) if kind == "lstm" else ViT()).to(device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.], device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=0., weight_decay=config["weight_decay"])
        def scheduled_lr(epoch):
            cycle = epoch // config["t0"]; position = epoch % config["t0"]
            peak = config["lr"] * config["gamma"] ** cycle
            if position < config["warmup"]:
                return peak * position / config["warmup"]
            phase = (position - config["warmup"]) / (config["t0"] - config["warmup"])
            return peak * (1 + math.cos(math.pi * phase)) / 2
        best, best_state, stale, history = float("inf"), None, 0, []
        for epoch in range(config["epochs"]):
            for group in optimizer.param_groups: group["lr"] = scheduled_lr(epoch)
            model.train(); train_loss = 0.
            for xb, yb in loaders["train"]:
                xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb); loss.backward()
                if kind == "lstm": torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.5)
                optimizer.step()
                train_loss += loss.item() * len(yb)
            model.eval(); val_loss = 0.
            validation_loss = nn.BCEWithLogitsLoss(reduction="sum")
            with torch.no_grad():
                for xb, yb in loaders["validation"]:
                    xb, yb = xb.to(device), yb.to(device); val_loss += validation_loss(model(xb), yb).item()
            train_loss /= len(loaders["train"].dataset); val_loss /= len(loaders["validation"].dataset)
            history.append({"model": kind, "seed": seed, "epoch": epoch + 1, "train_loss": train_loss, "validation_loss": val_loss})
            minimum_epoch = 15 if kind == "lstm" else 10
            if epoch > minimum_epoch and val_loss < best:
                best, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= config["patience"]:
                    break
        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); torch.save(best_state, CHECKPOINTS / f"{kind}_seed{seed}.pt")
        pd.DataFrame(history).to_csv(OUT / f"{kind}_seed{seed}_history.csv", index=False)
        model.eval()
        for name, idx in (("validation", va), ("test", te)):
            probs = []
            with torch.no_grad():
                for xb, _ in loaders[name]: probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel())
            probs = np.asarray(probs); rows.append(metric_row(y[idx], probs, kind, seed, name))
            predictions.extend({"model": kind, "seed": seed, "split": name, "row_index": int(i), "y_true": int(y[i]), "probability": float(p)} for i, p in zip(idx, probs))
    return rows, predictions


def save(rows, predictions, model, delta_mode, ids, names):
    metrics = pd.DataFrame(rows); preds = pd.DataFrame(predictions)
    lookup = pd.DataFrame({"row_index": np.arange(len(ids)), "pitcher_id": ids, "player_name": names})
    preds = preds.merge(lookup, on="row_index", how="left")
    metrics["delta_mode"] = delta_mode; preds["delta_mode"] = delta_mode
    summary = metrics.groupby(["model", "split"])[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])
    metrics.to_csv(OUT / f"{model}_{delta_mode}_metrics.csv", index=False)
    preds.to_csv(OUT / f"{model}_{delta_mode}_predictions.csv", index=False)
    with pd.ExcelWriter(OUT / f"{model}_{delta_mode}_results.xlsx") as writer:
        metrics.to_excel(writer, sheet_name="01_all_seeds", index=False)
        summary.to_excel(writer, sheet_name="02_mean_std")
    test = metrics.query("split == 'test'")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.errorbar(["F1", "ROC-AUC", "PR-AUC"], [test.f1.mean(), test.roc_auc.mean(), test.pr_auc.mean()], yerr=[test.f1.std(), test.roc_auc.std(), test.pr_auc.std()], fmt="o", capsize=5)
    ax.set_ylim(0, 1); ax.set_title(f"{model}: test mean +/- SD across seeds"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(OUT / f"{model}_{delta_mode}_summary.png", dpi=180); plt.close(fig)
    print(summary.to_string())


def main():
    a = args(); OUT.mkdir(parents=True, exist_ok=True); CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    z = np.load(DATA / f"tjs_sequences_{a.delta_mode}.npz", allow_pickle=True)
    X, y = z["X"], z["y"]
    rows, preds = train_logistic(X, y, a.seeds) if a.model == "logistic" else train_deep(X, y, a.model, a.seeds, a.epochs)
    save(rows, preds, a.model, a.delta_mode, z["pitcher_id"], z["player_name"])
    (OUT / f"{a.model}_{a.delta_mode}_run.json").write_text(json.dumps({"model": a.model, "delta_mode": a.delta_mode, "seeds": a.seeds, "shape": list(X.shape)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
