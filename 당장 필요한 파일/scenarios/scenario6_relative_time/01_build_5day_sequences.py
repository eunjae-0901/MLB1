"""시나리오 6: day 0 이전 100일을 20개의 5일 bin으로 재구성한다."""
from pathlib import Path
import json,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;WORK=HERE.parents[2];ROOT=WORK/"필요없는거"/"mlb_injury_project";BASE=HERE.parents[1]/"data";OUT=HERE/"data"
GROUPS=("FB","SI","CT","SL","CB","CH","SP")
FEATURES=["g_v_all","g_x_all","g_z_all","g_ext_all","g_spin_all","g_n_pitches"]+[f"g_{m}{g}" for g in GROUPS for m in ("pct_","v","x","z","ext","spin")]
def game_features(g):
 d=pd.DataFrame(index=g.index)
 for n,num,den in (("g_v_all","sum_v_all","nv_all"),("g_x_all","sum_x_all","nx_all"),("g_z_all","sum_z_all","nz_all"),("g_ext_all","sum_ext_all","next_all"),("g_spin_all","sum_spin_all","nspin_all")):d[n]=g[num]/g[den].replace(0,np.nan)
 d["g_n_pitches"]=g.n_pitches
 for p in GROUPS:
  d[f"g_pct_{p}"]=g[f"n_{p}"]/g.n_pitches.replace(0,np.nan)
  for m in ("v","x","z","ext","spin"):d[f"g_{m}{p}"]=g[f"sum_{m}{p}"]/g[f"n{m}{p}"].replace(0,np.nan)
 return d
def build(role):
 use=["player_id","game_date","is_start","n_pitches","sum_v_all","nv_all","sum_x_all","nx_all","sum_z_all","nz_all","sum_ext_all","next_all","sum_spin_all","nspin_all"]
 for p in GROUPS:use += [f"n_{p}"]+[f"sum_{m}{p}" for m in ("v","x","z","ext","spin")]+[f"n{m}{p}" for m in ("v","x","z","ext","spin")]
 g=pd.read_parquet(ROOT/"data/interim/pitcher_game/pitcher_game_role.parquet",columns=use);g=g[g.is_start if role=="starter" else ~g.is_start].copy();g["game_date"]=pd.to_datetime(g.game_date)
 gf=game_features(g);g=pd.concat([g[["player_id","game_date"]].reset_index(drop=True),gf.reset_index(drop=True)],axis=1).sort_values(["player_id","game_date"]);groups={k:v for k,v in g.groupby("player_id")}
 rows=pd.read_csv(BASE/f"{role}_dataset.csv");rows=rows[rows.label!=3].reset_index(drop=True);end=pd.to_datetime(rows.window_end_date);n=len(rows);X=np.full((n,20,len(FEATURES)),np.nan,np.float32);mask=np.zeros((n,20),np.uint8)
 for i,(pid,day) in enumerate(zip(rows.player_id,end)):
  z=groups.get(pid)
  if z is None:continue
  delta=(z.game_date-day).dt.days;z=z[(delta<=0)&(delta>=-99)].copy()
  if z.empty:continue
  z["bin"]=((z.game_date-day).dt.days+99)//5
  a=z.groupby("bin")[FEATURES].mean()
  for b,v in a.iterrows():X[i,int(b)]=v.to_numpy(np.float32);mask[i,int(b)]=1
 # train에서만 산출한 feature median으로 결측 대체; mask는 별도 보존
 split=rows.split.to_numpy();train=split=="train";med=np.nanmedian(X[train],axis=(0,1));inds=np.where(np.isnan(X));X[inds]=med[inds[2]]
 y=(rows.label.to_numpy()>0).astype(np.int8)
 OUT.mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/f"{role}_relative_5day.npz",X=X,mask=mask,y=y,split=split,player_id=rows.player_id.to_numpy(),window_end_date=rows.window_end_date.to_numpy(),il_start_date=rows.il_start_date.fillna("").to_numpy(),days_to_injury=rows.days_to_injury.to_numpy())
 (OUT/f"{role}_relative_5day_meta.json").write_text(json.dumps({"role":role,"days":100,"bin_days":5,"n_bins":20,"feature_names":FEATURES,"shape":list(X.shape)},ensure_ascii=False,indent=2),encoding="utf-8")
 print(role,X.shape,dict(pd.Series(split).value_counts()),"observed bins",mask.mean())
def main():
 for r in ("bullpen","starter"):build(r)
if __name__=="__main__":main()
