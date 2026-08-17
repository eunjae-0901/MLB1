"""Raw Statcast에서 구종 내 runner-context gap을 만들고 window에 연결한다."""
from pathlib import Path
import duckdb,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]/"필요없는거"/"mlb_injury_project";BASE=HERE.parents[1]/"data";OUT=HERE/"data"
PITCH="""CASE pitch_type WHEN 'FF' THEN 'FB' WHEN 'FA' THEN 'FB' WHEN 'SI' THEN 'SI' WHEN 'FT' THEN 'SI' WHEN 'FC' THEN 'CT' WHEN 'SL' THEN 'SL' WHEN 'ST' THEN 'SL' WHEN 'SV' THEN 'SL' WHEN 'CU' THEN 'CB' WHEN 'KC' THEN 'CB' WHEN 'CS' THEN 'CB' WHEN 'CH' THEN 'CH' WHEN 'FS' THEN 'SP' WHEN 'FO' THEN 'SP' ELSE 'OTHER' END"""
MET={"velocity":"release_speed","spin":"release_spin_rate","extension":"release_extension","pfx_x":"pfx_x*12","pfx_z":"pfx_z*12"}
def game_gap():
    glob=str(ROOT/"data/raw/statcast/statcast_*.parquet").replace("\\","/")
    av=",".join([f"avg({e}) FILTER(WHERE NOT has_runner) nr_{k},avg({e}) FILTER(WHERE has_runner) r_{k}" for k,e in MET.items()])
    gaps=",".join([f"sum(({ 'nr_'+k}-{ 'r_'+k})*(n_nr+n_r)) FILTER(WHERE n_nr>0 AND n_r>0)/nullif(sum(n_nr+n_r) FILTER(WHERE n_nr>0 AND n_r>0),0) runner_gap_{k}" for k in MET])
    q=f"""WITH r AS(SELECT pitcher::BIGINT player_id,game_pk,game_date::DATE game_date,{PITCH} pg,(on_1b IS NOT NULL OR on_2b IS NOT NULL OR on_3b IS NOT NULL) has_runner,release_speed,release_spin_rate,release_extension,pfx_x,pfx_z FROM read_parquet('{glob}') WHERE game_type='R'),a AS(SELECT player_id,game_pk,game_date,pg,count(*) FILTER(WHERE has_runner) n_r,count(*) FILTER(WHERE NOT has_runner) n_nr,{av} FROM r WHERE pg!='OTHER' GROUP BY ALL) SELECT player_id,game_pk,game_date,sum(n_r) runner_pitch_count,sum(n_nr) no_runner_pitch_count,{gaps} FROM a GROUP BY ALL"""
    return duckdb.sql(q).df()
def attach(role,gaps):
    base=pd.read_csv(BASE/f"{role}_dataset.csv"); games=pd.read_parquet(ROOT/"data/interim/pitcher_game/pitcher_game_role.parquet",columns=["player_id","game_pk","game_date","is_start"])
    games["game_date"]=pd.to_datetime(games.game_date);gaps["game_date"]=pd.to_datetime(gaps.game_date);g=games.merge(gaps,on=["player_id","game_pk","game_date"],how="left")
    g=g[g.is_start if role=="starter" else ~g.is_start].sort_values(["player_id","game_date"]);groups={k:v for k,v in g.groupby("player_id")}
    cols=[f"runner_gap_{k}" for k in MET];out=np.full((len(base),len(cols)+2),np.nan);dates=pd.to_datetime(base.window_end_date)
    for i,(pid,end) in enumerate(zip(base.player_id,dates)):
        x=groups.get(pid)
        if x is None:continue
        ds=x.game_date.to_numpy();j=np.searchsorted(ds,np.datetime64(end),side="right")
        if role=="starter":w=x.iloc[max(0,j-3):j]
        else:
            lo=np.datetime64(end-pd.Timedelta(days=13));a=np.searchsorted(ds,lo,side="left");w=x.iloc[a:j]
        nr=w.no_runner_pitch_count.sum();rr=w.runner_pitch_count.sum();out[i,-2:]=[rr,nr]
        if rr<5 or nr<5:continue
        for z,c in enumerate(cols):
            valid=w[c].notna();weight=(w.runner_pitch_count+w.no_runner_pitch_count)
            if valid.any():out[i,z]=np.average(w.loc[valid,c],weights=weight[valid])
    for z,c in enumerate(cols+["runner_pitch_count","no_runner_pitch_count"]):base[c]=out[:,z]
    OUT.mkdir(parents=True,exist_ok=True);base.to_parquet(OUT/f"{role}_runner_gap.parquet",index=False);print(role,base[cols].notna().mean().to_dict())
def main():
    g=game_gap();OUT.mkdir(parents=True,exist_ok=True);g.to_parquet(OUT/"game_runner_gap.parquet",index=False)
    for r in ("bullpen","starter"):attach(r,g)
if __name__=="__main__":main()
