"""시나리오 16: 동일 표본·feature로 7/14/30일 arm injury horizon 비교."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np,pandas as pd,xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];ROOT=HERE.parents[2]/"필요없는거"/"mlb_injury_project";OUT=HERE/"results"
p=PROJECT/"scenarios"/"scenario1_case_control"/"02_xgboost.py";s=importlib.util.spec_from_file_location("c",p);C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
def next_arm_days(df,ep):
    dates={int(k):np.sort(g.il_start_date.to_numpy(dtype="datetime64[D]")) for k,g in ep[ep.injury_class_strict.isin([1,2])].dropna(subset=["player_id"]).groupby(ep.player_id.astype("Int64"))}
    out=np.full(len(df),np.inf); wd=pd.to_datetime(df.window_end_date).to_numpy(dtype="datetime64[D]")
    for i,(pid,day) in enumerate(zip(df.player_id,wd)):
        a=dates.get(int(pid))
        if a is None:continue
        j=np.searchsorted(a,day,side="right")
        if j<len(a):out[i]=(a[j]-day).astype(int)
    return out
def run(role,horizon,ep):
    splits=C.load_splits(role,"baseline")
    for k,d in splits.items():d["label"]=(next_arm_days(d,ep)<=horizon).astype("int8")
    x,y,f=C.preprocess(splits);w=compute_sample_weight("balanced",y["train"]);params,t=C.tune(x["train"],y["train"],x["val"],y["val"],w,6,-1)
    m=xgb.XGBClassifier(**C.base_params(-1,**params));m.fit(x["train"],y["train"],sample_weight=w,eval_set=[(x["val"],y["val"])],verbose=False)
    pv=m.predict_proba(x["val"])[:,1];th=C.choose_threshold(y["val"],pv);rows=[]
    for sp in ("val","test"):
        pr=pv if sp=="val" else m.predict_proba(x["test"])[:,1]
        for tn,cut in (("0.5",.5),("validation_selected",th)):rows.append({"role":role,"horizon_days":horizon,"split":sp,"threshold_type":tn,"threshold":cut,"positive_rate":y[sp].mean(),**C.metrics(y[sp],pr,cut)})
    t.insert(0,"role",role);t.insert(1,"horizon_days",horizon);return rows,t
def main():
    ep=pd.read_parquet(ROOT/"data/interim/injury_episodes/injury_episodes.parquet");rows=[];tu=[]
    for r in ("bullpen","starter"):
        for h in (7,14,30):a,b=run(r,h,ep);rows+=a;tu.append(b)
    OUT.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows)
    with pd.ExcelWriter(OUT/"scenario16_results.xlsx") as w:df.to_excel(w,sheet_name="01_by_horizon",index=False);pd.concat(tu).to_excel(w,sheet_name="02_tuning",index=False)
    df.to_csv(OUT/"scenario16_metrics.csv",index=False);print(df[(df.split=="test")&(df.threshold_type=="validation_selected")].to_string(index=False))
if __name__=="__main__":main()
