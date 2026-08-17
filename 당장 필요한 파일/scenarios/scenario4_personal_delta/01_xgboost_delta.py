"""시나리오 4: train 시기의 선수별 baseline 대비 구위 변화량 실험."""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
COMMON_PATH=PROJECT/"scenarios"/"scenario1_case_control"/"02_xgboost.py"
spec=importlib.util.spec_from_file_location("common",COMMON_PATH); C=importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
OUT=HERE/"results"; MODELS=OUT/"models"

def add_delta(splits):
    train=splits["train"]; w=[c for c in train.columns if c.startswith("w_")]
    player=train.groupby("player_id")[w].mean()
    league=train[w].mean()
    out={}
    for name,df in splits.items():
        base=df[["player_id"]].join(player,on="player_id")[w].fillna(league)
        base.index=df.index
        d=df.copy()
        for col in w: d[f"delta_{col}"]=d[col]-base[col]
        out[name]=d
    return out,w

def run(role,feature_set,trials):
    splits=C.load_splits(role,"baseline"); splits,w=add_delta(splits)
    if feature_set=="delta_only":
        drop=[c for c in w]
        for s in splits: splits[s]=splits[s].drop(columns=drop)
    x,y,features=C.preprocess(splits)
    weights=compute_sample_weight("balanced",y["train"])
    params,tuning=C.tune(x["train"],y["train"],x["val"],y["val"],weights,trials,-1)
    model=xgb.XGBClassifier(**C.base_params(-1,**params))
    model.fit(x["train"],y["train"],sample_weight=weights,eval_set=[(x["val"],y["val"])],verbose=False)
    pv=model.predict_proba(x["val"])[:,1]; th=C.choose_threshold(y["val"],pv)
    rows=[]
    for split in ("val","test"):
        prob=pv if split=="val" else model.predict_proba(x["test"])[:,1]
        for tn,t in (("0.5",.5),("validation_selected",th)):
            rows.append({"role":role,"feature_set":feature_set,"split":split,"threshold_type":tn,"threshold":t,
                         "n_features":len(features),**C.metrics(y[split],prob,t)})
    MODELS.mkdir(parents=True,exist_ok=True); model.save_model(MODELS/f"{role}_{feature_set}.json")
    tuning.insert(0,"role",role); tuning.insert(1,"feature_set",feature_set)
    return rows,tuning

def main():
    p=argparse.ArgumentParser(); p.add_argument("--trials",type=int,default=6); a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); rows=[]; tunes=[]
    for role in ("bullpen","starter"):
        for fs in ("absolute_plus_delta","delta_only"):
            r,t=run(role,fs,a.trials); rows+=r; tunes.append(t)
    m=pd.DataFrame(rows); t=pd.concat(tunes,ignore_index=True)
    with pd.ExcelWriter(OUT/"scenario4_results.xlsx") as w:
        m.to_excel(w,sheet_name="01_all_metrics",index=False); t.to_excel(w,sheet_name="02_tuning",index=False)
    m.to_csv(OUT/"scenario4_metrics.csv",index=False)
    print(m[(m.split=="test")&(m.threshold_type=="validation_selected")].to_string(index=False))
if __name__=="__main__": main()
