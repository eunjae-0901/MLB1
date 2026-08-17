"""시나리오 13: 기준일 이전 부상 이력 feature 구축 및 XGBoost 실험."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np,pandas as pd,xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[1]; ROOT=HERE.parents[2]/"필요없는거"/"mlb_injury_project"; OUT=HERE/"results"
p=PROJECT/"scenarios"/"scenario1_case_control"/"02_xgboost.py";s=importlib.util.spec_from_file_location("c",p);C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
H=["past_IL_count","past_shoulder_IL_count","past_elbow_IL_count","days_since_last_IL","past_365d_IL_days","previous_arm_injury"]
def add_history(df,episodes):
    d=df.copy();d["window_end_date"]=pd.to_datetime(d["window_end_date"]); vals={c:np.zeros(len(d)) for c in H};vals["days_since_last_IL"][:]=9999
    pos=pd.Series(np.arange(len(d)),index=d.index)
    epgroups={int(k):g.sort_values("il_start_date") for k,g in episodes.dropna(subset=["player_id"]).groupby(episodes.player_id.astype("Int64"))}
    for pid,idx in d.groupby("player_id").groups.items():
        g=epgroups.get(int(pid))
        if g is None: continue
        starts=g.il_start_date.to_numpy(dtype="datetime64[D]"); ends=g.il_end_date.fillna(g.il_start_date).to_numpy(dtype="datetime64[D]"); cls=g.injury_class_strict.to_numpy()
        for ix in idx:
            day=np.datetime64(d.at[ix,"window_end_date"].date()); mask=starts<day; loc=pos[ix]
            if not mask.any(): continue
            vals["past_IL_count"][loc]=mask.sum();vals["past_shoulder_IL_count"][loc]=((cls==1)&mask).sum();vals["past_elbow_IL_count"][loc]=((cls==2)&mask).sum()
            vals["days_since_last_IL"][loc]=(day-starts[mask].max()).astype(int);vals["previous_arm_injury"][loc]=(((cls==1)|(cls==2))&mask).any()
            lo=day-np.timedelta64(365,"D"); overlap=np.maximum(np.timedelta64(0,"D"),np.minimum(ends[mask],day)-np.maximum(starts[mask],lo))
            vals["past_365d_IL_days"][loc]=sum(x.astype(int) for x in overlap)
    for c,v in vals.items():d[c]=v
    return d
def run(role,fs,episodes):
    splits=C.load_splits(role,"baseline");splits={k:add_history(v,episodes) for k,v in splits.items()}
    if fs=="history_only":
        keep=set(H)|C.ID_COLS|set(C.CATEGORICAL_COLS)|{"label","split"};splits={k:v[[c for c in v.columns if c in keep]] for k,v in splits.items()}
    x,y,f=C.preprocess(splits);w=compute_sample_weight("balanced",y["train"]);params,t=C.tune(x["train"],y["train"],x["val"],y["val"],w,6,-1)
    m=xgb.XGBClassifier(**C.base_params(-1,**params));m.fit(x["train"],y["train"],sample_weight=w,eval_set=[(x["val"],y["val"])],verbose=False)
    pv=m.predict_proba(x["val"])[:,1];th=C.choose_threshold(y["val"],pv);rows=[]
    for sp in ("val","test"):
        pr=pv if sp=="val" else m.predict_proba(x["test"])[:,1]
        for tn,cut in (("0.5",.5),("validation_selected",th)):rows.append({"role":role,"feature_set":fs,"split":sp,"threshold_type":tn,"threshold":cut,"n_features":len(f),**C.metrics(y[sp],pr,cut)})
    t.insert(0,"role",role);t.insert(1,"feature_set",fs);return rows,t
def main():
    ep=pd.read_parquet(ROOT/"data/interim/injury_episodes/injury_episodes.parquet");rows=[];tu=[]
    for r in ("bullpen","starter"):
        for fs in ("history_only","base_plus_history"):a,b=run(r,fs,ep);rows+=a;tu.append(b)
    OUT.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows)
    with pd.ExcelWriter(OUT/"scenario13_results.xlsx") as w:df.to_excel(w,sheet_name="01_metrics",index=False);pd.concat(tu).to_excel(w,sheet_name="02_tuning",index=False)
    df.to_csv(OUT/"scenario13_metrics.csv",index=False);print(df[(df.split=="test")&(df.threshold_type=="validation_selected")].to_string(index=False))
if __name__=="__main__":main()
