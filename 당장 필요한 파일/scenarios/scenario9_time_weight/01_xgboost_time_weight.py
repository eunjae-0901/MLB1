"""시나리오 9: class weight에 부상 임박도 가중치를 곱한다."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
HERE=Path(__file__).resolve().parent; PROJECT=HERE.parents[1]; OUT=HERE/"results"
p=PROJECT/"scenarios"/"scenario1_case_control"/"02_xgboost.py"
s=importlib.util.spec_from_file_location("common",p); C=importlib.util.module_from_spec(s); s.loader.exec_module(C)
def run(role,method):
    splits=C.load_splits(role,"baseline"); x,y,f=C.preprocess(splits)
    w=compute_sample_weight("balanced",y["train"])
    if method=="class_x_time":
        days=splits["train"]["days_to_injury"].fillna(14).clip(0,14).to_numpy()
        factor=np.where(y["train"].to_numpy()==1,1+(14-days)/14,1)
        w=w*factor
    params,t=C.tune(x["train"],y["train"],x["val"],y["val"],w,6,-1)
    m=xgb.XGBClassifier(**C.base_params(-1,**params)); m.fit(x["train"],y["train"],sample_weight=w,eval_set=[(x["val"],y["val"])],verbose=False)
    pv=m.predict_proba(x["val"])[:,1]; th=C.choose_threshold(y["val"],pv); rows=[]
    for split in ("val","test"):
        prob=pv if split=="val" else m.predict_proba(x["test"])[:,1]
        for tn,cut in (("0.5",.5),("validation_selected",th)):
            rows.append({"role":role,"weight_method":method,"split":split,"threshold_type":tn,"threshold":cut,**C.metrics(y[split],prob,cut)})
    t.insert(0,"role",role);t.insert(1,"weight_method",method);return rows,t
def main():
    OUT.mkdir(parents=True,exist_ok=True); rows=[]; tunes=[]
    for r in ("bullpen","starter"):
        for m in ("class_only","class_x_time"):
            a,b=run(r,m);rows+=a;tunes.append(b)
    df=pd.DataFrame(rows);tu=pd.concat(tunes)
    with pd.ExcelWriter(OUT/"scenario9_results.xlsx") as w: df.to_excel(w,sheet_name="01_metrics",index=False);tu.to_excel(w,sheet_name="02_tuning",index=False)
    df.to_csv(OUT/"scenario9_metrics.csv",index=False);print(df[(df.split=="test")&(df.threshold_type=="validation_selected")].to_string(index=False))
if __name__=="__main__":main()
