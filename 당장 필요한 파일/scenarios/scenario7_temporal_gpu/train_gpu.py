"""GPU용 LSTM/ViT 반복 seed 학습. 노트북에서 호출한다."""
import argparse,json,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import *
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
HERE=Path(__file__).resolve().parent;DATA=HERE.parent/"scenario6_relative_time"/"data";OUT=HERE/"results"
def seedall(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
class LSTM(nn.Module):
 def __init__(self,p):super().__init__();self.r=nn.LSTM(p,64,batch_first=True);self.o=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
 def forward(self,x,m):y,_=self.r(x);idx=m.sum(1).long().clamp(min=1)-1;return self.o(y[torch.arange(len(y),device=y.device),idx]).squeeze(1)
class ViT(nn.Module):
 def __init__(self):
  super().__init__();import timm;self.m=timm.create_model("vit_base_patch16_224",pretrained=True,num_classes=1)
 def forward(self,x,m):return self.m(torch.nn.functional.interpolate(x.unsqueeze(1).repeat(1,3,1,1),size=(224,224))).squeeze(1)
def metrics(y,p,t):
 z=(p>=t);return dict(roc_auc=roc_auc_score(y,p),pr_auc=average_precision_score(y,p),f1=f1_score(y,z),precision=precision_score(y,z,zero_division=0),recall=recall_score(y,z),mcc=matthews_corrcoef(y,z),brier=brier_score_loss(y,p))
def threshold(y,p):
 a,b,t=precision_recall_curve(y,p);f=2*a[:-1]*b[:-1]/(a[:-1]+b[:-1]+1e-12);return float(t[np.argmax(f)])
def run(role,kind,seed):
 seedall(seed);z=np.load(DATA/f"{role}_relative_5day.npz",allow_pickle=True);X=z["X"].astype("float32");M=z["mask"].astype("float32");y=z["y"].astype("float32");sp=z["split"].astype(str)
 tr=sp=="train";mean=X[tr].mean((0,1));std=X[tr].std((0,1))+1e-6;X=(X-mean)/std
 dev=torch.device("cuda");model=(LSTM(X.shape[-1]) if kind=="lstm" else ViT()).to(dev);pos=y[tr].sum();pw=torch.tensor([(tr.sum()-pos)/pos],device=dev);loss=nn.BCEWithLogitsLoss(pos_weight=pw);opt=torch.optim.AdamW(model.parameters(),lr=1e-3 if kind=="lstm" else 2e-5,weight_decay=1e-4)
 def loader(mask,shuffle=False):return DataLoader(TensorDataset(torch.from_numpy(X[mask]),torch.from_numpy(M[mask]),torch.from_numpy(y[mask])),batch_size=256 if kind=="lstm" else 32,shuffle=shuffle)
 best=None;pat=0
 for epoch in range(50):
  model.train()
  for a,b,c in loader(tr,True):a,b,c=a.to(dev),b.to(dev),c.to(dev);opt.zero_grad();q=model(a,b);l=loss(q,c);l.backward();opt.step()
  pv=predict(model,loader(sp=="val"),dev);score=average_precision_score(y[sp=="val"],pv)
  if best is None or score>best[0]:best=(score,{k:v.cpu().clone() for k,v in model.state_dict().items()});pat=0
  else:pat+=1
  if pat>=7:break
 model.load_state_dict(best[1]);pv=predict(model,loader(sp=="val"),dev);th=threshold(y[sp=="val"],pv);rows=[]
 for split in ("val","test"):
  mask=sp==split;p=pv if split=="val" else predict(model,loader(mask),dev)
  for tn,t in (("0.5",.5),("validation_selected",th)):rows.append(dict(role=role,model=kind,seed=seed,split=split,threshold_type=tn,threshold=t,**metrics(y[mask],p,t)))
 return rows
@torch.no_grad()
def predict(m,l,d):
 m.eval();return np.concatenate([torch.sigmoid(m(a.to(d),b.to(d))).cpu().numpy() for a,b,_ in l])
def main():
 p=argparse.ArgumentParser();p.add_argument("--model",choices=["lstm","vit"],required=True);p.add_argument("--seeds",nargs="+",type=int,default=[11,22,33,44,55]);a=p.parse_args();OUT.mkdir(exist_ok=True);rows=[]
 for r in ("bullpen","starter"):
  for s in a.seeds:rows+=run(r,a.model,s)
 d=pd.DataFrame(rows);d.to_csv(OUT/f"{a.model}_all_seeds.csv",index=False);d.to_excel(OUT/f"{a.model}_results.xlsx",index=False);print(d.groupby(["role","split","threshold_type"])[["roc_auc","pr_auc","f1","mcc","brier"]].agg(["mean","std"]))
if __name__=="__main__":main()
