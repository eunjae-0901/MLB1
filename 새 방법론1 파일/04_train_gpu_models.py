"""GPU LSTM/ViT multi-task training for 100-day injury risk and days-to-IL."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from modeling_common import DATA, RESULTS, choose_threshold, metric_row


class InjuryDataset(Dataset):
    def __init__(self, x, mask, y, days, weight, indices, kind):
        self.x, self.mask, self.y, self.days, self.weight = x, mask, y, days, weight
        self.indices, self.kind = np.asarray(indices), kind
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        x = torch.from_numpy(self.x[j])
        m = torch.from_numpy(self.mask[j].astype("float32"))
        if self.kind in {"lstm", "resnet"}: x = torch.cat([x, m[:, None]], dim=1)
        return x, m, torch.tensor(self.y[j]), torch.tensor(self.days[j]), torch.tensor(self.weight[j]), j


class LSTMModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.rnn = nn.LSTM(n_features + 1, 128, num_layers=2, batch_first=True,
                           dropout=.25, bidirectional=True)
        self.shared = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(.3))
        self.cls, self.reg = nn.Linear(128, 1), nn.Linear(128, 1)
    def forward(self, x, mask):
        h, _ = self.rnn(x)
        z = (h * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.shared(z); return self.cls(z).squeeze(1), F.softplus(self.reg(z).squeeze(1))


class ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        d = self.backbone.num_features
        self.cls, self.reg = nn.Linear(d, 1), nn.Linear(d, 1)
    def forward(self, x, mask):
        image = F.interpolate(x[:, None], size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        z = self.backbone(image); return self.cls(z).squeeze(1), F.softplus(self.reg(z).squeeze(1))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, 1),
                                 nn.SiLU(), nn.Conv1d(hidden, channels, 1), nn.Sigmoid())
    def forward(self, x): return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=.20):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(8, channels), SqueezeExcitation(channels))
        self.activation = nn.SiLU()
    def forward(self, x): return self.activation(x + self.body(x))


class ResNet1DModel(nn.Module):
    """Compact residual TCN for short, sparse 20-bin clinical time series."""
    def __init__(self, n_features, channels=96):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_features + 1, channels, 1, bias=False),
                                  nn.GroupNorm(8, channels), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(channels, d) for d in (1, 2, 4, 8)])
        self.shared = nn.Sequential(nn.Linear(channels * 2, 128), nn.SiLU(), nn.Dropout(.35))
        self.cls, self.reg = nn.Linear(128, 1), nn.Linear(128, 1)
    def forward(self, x, mask):
        h = self.blocks(self.stem(x.transpose(1, 2)))
        m = mask[:, None].bool()
        mean = (h * m).sum(2) / m.sum(2).clamp(min=1)
        maximum = h.masked_fill(~m, torch.finfo(h.dtype).min).max(2).values
        z = self.shared(torch.cat([mean, maximum], dim=1))
        return self.cls(z).squeeze(1), F.softplus(self.reg(z).squeeze(1))


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ids, prob, days = [], [], []
    for x, mask, _, _, _, idx in loader:
        logits, reg = model(x.to(device), mask.to(device))
        ids.extend(idx.numpy()); prob.extend(torch.sigmoid(logits).cpu().numpy()); days.extend((100*reg).cpu().numpy())
    order = np.argsort(ids)
    return np.asarray(ids)[order], np.asarray(prob)[order], np.asarray(days)[order]


def main(kind, seed, epochs):
    if not torch.cuda.is_available(): raise SystemExit("CUDA GPU가 필요합니다.")
    torch.manual_seed(seed); np.random.seed(seed); device = torch.device("cuda")
    z = np.load(DATA / "sequences_100d_5d.npz"); x = z["X"].astype("float32")[:, ::-1].copy(); mask = z["mask"][:, ::-1].copy()
    meta = pd.read_parquet(DATA / "sequence_metadata.parquet")
    labeled = meta.target_100d.notna().to_numpy(); train = (meta.split.eq("train").to_numpy() & labeled); val = (meta.split.eq("validation").to_numpy() & labeled)
    # Fit normalization on Train only.
    mean = np.nanmean(x[train], axis=(0, 1)); std = np.nanstd(x[train], axis=(0, 1)); std[std < 1e-6] = 1
    x = np.nan_to_num((x - mean) / std, nan=0, posinf=0, neginf=0).astype("float32")
    static_mean = static_std = None
    if kind == "resnet":
        # Add within-pitcher 100-day deviations and static prior-history covariates.
        valid = mask[:, :, None].astype("float32")
        personal_mean = (x * valid).sum(1, keepdims=True) / valid.sum(1, keepdims=True).clip(min=1)
        delta = (x - personal_mean) * valid
        static = np.column_stack([
            meta.past_arm_il_count.fillna(0).to_numpy(float),
            np.log1p(meta.days_since_last_arm_il.fillna(3650).clip(lower=0).to_numpy(float)),
            meta.age.fillna(meta.loc[train, "age"].median()).to_numpy(float),
        ])
        static_mean, static_std = static[train].mean(0), static[train].std(0)
        static_std[static_std < 1e-6] = 1
        static = ((static - static_mean) / static_std).astype("float32")
        x = np.concatenate([x, delta, np.repeat(static[:, None], x.shape[1], axis=1)], axis=2)
    y = meta.target_100d.fillna(0).to_numpy("float32"); days = meta.regression_days.fillna(0).to_numpy("float32") / 100
    event_w = meta.event_weight.fillna(1).to_numpy("float32")
    pos_factor = float((y[train] == 0).sum() / event_w[train & (y == 1)].sum())
    weight = np.where(y == 1, event_w * pos_factor, 1).astype("float32")
    def loader(indices, shuffle=False):
        return DataLoader(InjuryDataset(x, mask, y, days, weight, indices, kind),
                          batch_size=256 if kind in {"lstm", "resnet"} else 16, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    tr_idx, va_idx = np.flatnonzero(train), np.flatnonzero(val)
    model = ({"lstm": lambda: LSTMModel(x.shape[2]), "vit": ViTModel,
              "resnet": lambda: ResNet1DModel(x.shape[2])}[kind]()).to(device)
    learning_rate = {"lstm": 3e-4, "vit": 2e-5, "resnet": 2e-4}[kind]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=3e-4 if kind == "resnet" else 1e-4)
    best_score, best_state, patience = -1., None, 12
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.
        for xb, mb, yb, db, wb, _ in loader(tr_idx, True):
            xb, mb, yb, db, wb = [q.to(device) for q in (xb, mb, yb, db, wb)]
            optimizer.zero_grad(); logits, reg = model(xb, mb)
            target = yb * .95 + .025 if kind == "resnet" else yb
            cls = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * wb).mean()
            positive = yb.eq(1); reg_loss = F.smooth_l1_loss(reg[positive], db[positive]) if positive.any() else cls * 0
            loss = cls + .30 * reg_loss; loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += loss.item() * len(yb)
        _, vp, _ = predict(model, loader(va_idx), device); score = average_precision_score(y[va_idx], vp)
        history.append({"epoch": epoch, "loss": total/max(len(tr_idx), 1), "val_pr_auc": score})
        if score > best_score: best_score, best_state, wait = score, copy.deepcopy(model.state_dict()), patience
        else:
            wait -= 1
            if wait == 0: break
    model.load_state_dict(best_state)
    _, val_p, _ = predict(model, loader(va_idx), device)
    calibrator = None
    if kind == "resnet":
        clipped = np.clip(val_p, 1e-6, 1-1e-6)
        calibrator = LogisticRegression(C=.5).fit(np.log(clipped/(1-clipped)).reshape(-1, 1), y[va_idx])
        val_p = calibrator.predict_proba(np.log(clipped/(1-clipped)).reshape(-1, 1))[:, 1]
    torch.save({"model": best_state, "mean": mean, "std": std,
                "static_mean": static_mean, "static_std": static_std,
                "calibration_coef": None if calibrator is None else calibrator.coef_,
                "calibration_intercept": None if calibrator is None else calibrator.intercept_},
               RESULTS/f"{kind}_seed{seed}.pt")
    threshold = choose_threshold(y[va_idx], val_p)
    rows, predictions = [], []
    cohorts = {"validation_all": meta.split.eq("validation"), "test_all": meta.split.eq("test"),
        "test_seen": meta.split.eq("test") & meta.evaluation_cohort.eq("seen_player"),
        "test_new": meta.split.eq("test") & meta.evaluation_cohort.eq("new_player"),
        "test_bullpen": meta.split.eq("test") & meta.role.eq("bullpen"),
        "test_starter": meta.split.eq("test") & meta.role.eq("starter")}
    for cohort, series in cohorts.items():
        idx = np.flatnonzero(series.to_numpy() & labeled); ids, p, pred_days = predict(model, loader(idx), device)
        if calibrator is not None:
            clipped = np.clip(p, 1e-6, 1-1e-6)
            p = calibrator.predict_proba(np.log(clipped/(1-clipped)).reshape(-1, 1))[:, 1]
        row = metric_row(y[ids], p, threshold, model=kind, seed=seed, cohort=cohort)
        positive = y[ids] == 1; row["regression_mae_days"] = float(mean_absolute_error(days[ids][positive]*100, pred_days[positive])) if positive.any() else np.nan
        rows.append(row)
        predictions.extend({"row_index":int(j),"probability":float(pp),"predicted_days":float(dd),"cohort":cohort} for j,pp,dd in zip(ids,p,pred_days))
    pd.DataFrame(rows).to_csv(RESULTS/f"{kind}_seed{seed}_metrics.csv",index=False)
    pd.DataFrame(predictions).to_csv(RESULTS/f"{kind}_seed{seed}_predictions.csv",index=False)
    pd.DataFrame(history).to_csv(RESULTS/f"{kind}_seed{seed}_history.csv",index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=["lstm","vit","resnet"],required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=100); a=p.parse_args()
    main(a.model,a.seed,a.epochs)
