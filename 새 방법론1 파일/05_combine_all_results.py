"""Combine CPU and returned GPU metrics into publication-ready tables."""
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent; OUT=HERE/"results"
files=[OUT/"xgboost_weighted_metrics.csv",OUT/"stack_ensemble_metrics.csv",*OUT.glob("lstm_seed*_metrics.csv"),
       *OUT.glob("vit_seed*_metrics.csv"),*OUT.glob("resnet_seed*_metrics.csv"),*OUT.glob("gbm_seed*_metrics.csv")]
frames=[pd.read_csv(f) for f in files if f.exists()]
if not frames: raise SystemExit("No metric files found.")
all_metrics=pd.concat(frames,ignore_index=True)
all_metrics.to_csv(OUT/"all_models_metrics.csv",index=False)
keys=["model","cohort"]
summary=all_metrics.groupby(keys).agg(n_runs=("pr_auc","size"),roc_auc_mean=("roc_auc","mean"),roc_auc_sd=("roc_auc","std"),pr_auc_mean=("pr_auc","mean"),pr_auc_sd=("pr_auc","std"),f1_mean=("f1","mean"),recall_mean=("recall","mean"),precision_mean=("precision","mean"),brier_mean=("brier","mean")).reset_index()
summary.to_csv(OUT/"all_models_summary.csv",index=False)
try:
    import matplotlib.pyplot as plt
    test=summary[summary.cohort.eq("test_all")]
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(test.model,test.pr_auc_mean,yerr=test.pr_auc_sd.fillna(0),capsize=4)
    ax.set(ylabel="PR-AUC",title="Final Test model comparison"); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(OUT/"all_models_test_pr_auc.png",dpi=200); plt.close(fig)
except ImportError:
    pass
print(summary.to_string(index=False))
