"""시나리오 5 누적 비교: base → +delta → +runner gap."""
import importlib.util
from pathlib import Path
import pandas as pd,xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1];OUT=HERE/"results"
p=PROJECT/"scenarios"/"scenario1_case_control"/"02_xgboost.py";s=importlib.util.spec_from_file_location("c",p);C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
G=[f"runner_gap_{x}" for x in ("velocity","spin","extension","pfx_x","pfx_z")]+["runner_pitch_count","no_runner_pitch_count"]
def load(role):
 d=pd.read_parquet(HERE/"data"/f"{role}_runner_gap.parquet");d=d[d.label!=3].copy();d.label=(d.label>0).astype("int8")
 for c in ("window_end_date","il_start_date"):d[c]=pd.to_datetime(d[c],errors="coerce")
 return {k:d[d.split==k].reset_index(drop=True) for k in ("train","val","test")}
def delta(sp):
 w=[c for c in sp["train"] if c.startswith("w_")];b=sp["train"].groupby("player_id")[w].mean();league=sp["train"][w].mean()
 for k,d in sp.items():
  z=d[["player_id"]].join(b,on="player_id")[w].fillna(league);z.index=d.index
  for c in w:d[f"delta_{c}"]=d[c]-z[c]
 return sp
def run(role,fs):
 sp=delta(load(role))
 if fs=="base":sp={k:d.drop(columns=[c for c in d if c.startswith("delta_") or c in G]) for k,d in sp.items()}
 elif fs=="base_delta":sp={k:d.drop(columns=G) for k,d in sp.items()}
 x,y,f=C.preprocess(sp);w=compute_sample_weight("balanced",y["train"]);pa,t=C.tune(x["train"],y["train"],x["val"],y["val"],w,6,-1)
 m=xgb.XGBClassifier(**C.base_params(-1,**pa));m.fit(x["train"],y["train"],sample_weight=w,eval_set=[(x["val"],y["val"])],verbose=False);pv=m.predict_proba(x["val"])[:,1];th=C.choose_threshold(y["val"],pv);rows=[]
 for q in ("val","test"):
  pr=pv if q=="val" else m.predict_proba(x["test"])[:,1]
  for tn,cut in (("0.5",.5),("validation_selected",th)):rows.append({"role":role,"feature_set":fs,"split":q,"threshold_type":tn,"threshold":cut,"n_features":len(f),**C.metrics(y[q],pr,cut)})
 t.insert(0,"role",role);t.insert(1,"feature_set",fs);return rows,t
def main():
 rows=[];tu=[]
 for r in ("bullpen","starter"):
  for fs in ("base","base_delta","base_delta_runner_gap"):a,b=run(r,fs);rows+=a;tu.append(b)
 OUT.mkdir(exist_ok=True);d=pd.DataFrame(rows)
 with pd.ExcelWriter(OUT/"scenario5_results.xlsx") as w:d.to_excel(w,sheet_name="01_metrics",index=False);pd.concat(tu).to_excel(w,sheet_name="02_tuning",index=False)
 d.to_csv(OUT/"scenario5_metrics.csv",index=False);print(d[(d.split=="test")&(d.threshold_type=="validation_selected")].to_string(index=False))
if __name__=="__main__":main()
