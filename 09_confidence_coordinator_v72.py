#!/usr/bin/env python3
from __future__ import annotations
import argparse,itertools,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, roc_auc_score, average_precision_score
VARIABLES=['temp_c','humidity','light']; EXPERTS=['spike','drift','flat','dropout']

def args():
 p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--out',required=True); p.add_argument('--train-ratio',type=float,default=.50); p.add_argument('--val-ratio',type=float,default=.20); p.add_argument('--window',type=int,default=21); p.add_argument('--grid-step',type=float,default=.25); return p.parse_args()
def safe(df,c):
 x=pd.to_numeric(df[c],errors='coerce').astype(float); return x.interpolate(limit=3,limit_area='inside').ffill(limit=1).bfill(limit=1)
def robust(x,w):
 k=max(3,w//2); med=x.rolling(w,center=True,min_periods=k).median(); mad=(x-med).abs().rolling(w,center=True,min_periods=k).median(); sig=(1.4826*mad).replace(0,np.nan); sd=float(np.nanstd(x)); sig=sig.fillna(sd if sd>1e-9 else 1.); med=med.fillna(x.rolling(w,center=True,min_periods=1).median()).fillna(x.median()); return med,sig
def norm(a,q=.98):
 a=np.nan_to_num(np.asarray(a,float),nan=0,posinf=0,neginf=0); d=np.quantile(a,q) if len(a) else 1.; d=d if np.isfinite(d) and d>1e-12 else max(np.max(a),1.); return np.clip(a/d,0,1)
def confidences(df,w):
 s={e:np.zeros(len(df)) for e in EXPERTS}; half=max(3,w//2)
 for v in VARIABLES:
  x=safe(df,v); med,sig=robust(x,w); rz=((x-med).abs()/sig).to_numpy(); dx=x.diff().abs().fillna(0); dm,ds=robust(dx,w); dz=((dx-dm).abs()/ds).to_numpy(); s['spike']=np.maximum(s['spike'],norm(np.maximum(rz/4,dz/4)))
  short=x.rolling(max(5,w//2),min_periods=3).mean(); long=x.rolling(max(w*3,31),min_periods=w).mean(); gap=(short-long).abs().fillna(0); slope=(x-x.shift(w-1)).abs().div(max(1,w-1)).fillna(0); sd=max(float(np.nanstd(x)),1e-9); s['drift']=np.maximum(s['drift'],norm(gap/sd+slope*w/sd))
  rstd=x.rolling(w,min_periods=half).std(ddof=0).fillna(sd); low=np.clip(1-rstd/(.20*sd+1e-9),0,1); tiny=(x.diff().abs().fillna(np.inf)<max(1e-6,.01*sd)).astype(float); rep=tiny.rolling(w,min_periods=half).mean().fillna(0); s['flat']=np.maximum(s['flat'],norm(np.maximum(low.to_numpy(),rep.to_numpy()),.95)); s['dropout']=np.maximum(s['dropout'],np.minimum(1,.65*rep.to_numpy()+.35*norm(dz)))
 return pd.DataFrame({f'conf_{e}':s[e] for e in EXPERTS})
def labels(df):
 out={}
 for e in EXPERTS:
  c=f'label_global_{e}'
  if c in df: out[e]=pd.to_numeric(df[c],errors='coerce').fillna(0).astype(int).to_numpy()
  else:
   sub=[f'label_{v}_{e}' for v in VARIABLES if f'label_{v}_{e}' in df]; out[e]=(df[sub].max(axis=1).to_numpy()>0).astype(int) if sub else np.zeros(len(df),int)
 out['global']=pd.to_numeric(df['label_global'],errors='coerce').fillna(0).astype(int).to_numpy() if 'label_global' in df else np.maximum.reduce([out[e] for e in EXPERTS]); return out
def met(y,p,score=None):
 y=np.asarray(y,int); p=np.asarray(p,int); tp=int(((y==1)&(p==1)).sum()); tn=int(((y==0)&(p==0)).sum()); fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum()); prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0; f1=2*prec*rec/(prec+rec) if prec+rec else 0; fpr=fp/(fp+tn) if fp+tn else 0; spec=tn/(tn+fp) if tn+fp else 0; acc=(tp+tn)/max(1,len(y)); bal=(rec+spec)/2; mcc=float(matthews_corrcoef(y,p)) if len(np.unique(y))>1 and len(np.unique(p))>1 else 0.; r=.45*f1+.30*rec+.15*prec-.10*fpr; d=dict(tp=tp,tn=tn,fp=fp,fn=fn,accuracy=acc,precision=prec,recall=rec,f1=f1,fpr=fpr,fnr=fn/(fn+tp) if fn+tp else 0,specificity=spec,balanced_accuracy=bal,mcc=mcc,reward=r)
 if score is not None and len(np.unique(y))>1:
  try:d['roc_auc']=roc_auc_score(y,score);d['pr_auc']=average_precision_score(y,score)
  except: d['roc_auc']=np.nan;d['pr_auc']=np.nan
 return d
def best_th(score,y,idx):
 best=(.5,-1e9)
 for t in np.linspace(.05,.95,19):
  r=met(y[idx],score[idx]>=t)['reward']; best=(float(t),r) if r>best[1] else best
 return best[0]
def grids(step):
 vals=np.round(np.arange(0,1+1e-9,step),6)
 for w in itertools.product(vals,repeat=4):
  if abs(sum(w)-1)<1e-9: yield np.array(w,float)
def main():
 a=args(); out=Path(a.out);out.mkdir(parents=True,exist_ok=True); rows=[]; exrows=[]; cfg=[]
 for f in sorted(Path(a.data).glob('sensor_*_multivar_labeled.csv')):
  t0=time.perf_counter();df=pd.read_csv(f);df.columns=[c.lower() for c in df.columns]; C=confidences(df,a.window).to_numpy(); L=labels(df); n=len(df); ntr=int(n*a.train_ratio); nv=int(n*a.val_ratio); tr=np.arange(ntr); va=np.arange(ntr,min(n,ntr+nv)); te=np.arange(min(n,ntr+nv),n)
  ep={};ths={}
  for j,e in enumerate(EXPERTS): th=best_th(C[:,j],L[e],tr);ths[e]=th;ep[e]=(C[:,j]>=th).astype(int); exrows.append({'sensor':f.stem,'expert':e,'split':'test','threshold':th,**met(L[e][te],ep[e][te],C[te,j])})
  # generic single detector: max expert confidence, one threshold trained globally
  maxc=C.max(1); single_th=best_th(maxc,L['global'],tr); single=(maxc>=single_th).astype(int)
  orall=np.maximum.reduce([ep[e] for e in EXPERTS]); maj=(sum(ep[e] for e in EXPERTS)>=2).astype(int); equal=C.mean(1); equal_th=best_th(equal,L['global'],tr); equalp=(equal>=equal_th).astype(int)
  best={'r':-1e9,'w':np.ones(4)/4,'th':.5}
  for w in grids(a.grid_step):
   sc=C@w
   for th in np.linspace(.05,.95,19):
    r=met(L['global'][va],sc[va]>=th)['reward']
    if r>best['r']:best={'r':r,'w':w.copy(),'th':float(th)}
  score=C@best['w']; coord=(score>=best['th']).astype(int); hybrid=np.maximum(coord,(C.max(1)>=.90).astype(int))
  methods={'single_generic':(single,maxc),'or_all':(orall,orall),'majority':(maj,maj),'equal_confidence':(equalp,equal),'confidence_coordinator':(coord,score),'confidence_or_hybrid':(hybrid,np.maximum(score,C.max(1)))}
  for name,(p,sc) in methods.items(): rows.append({'sensor':f.stem,'method':name,'split':'test','n_test':len(te),**met(L['global'][te],p[te],sc[te])})
  cfg.append({'sensor':f.stem,'n':n,'n_train':len(tr),'n_val':len(va),'n_test':len(te),'single_threshold':single_th,'equal_threshold':equal_th,'coord_threshold':best['th'],'coord_val_reward':best['r'],**{f'w_{e}':best['w'][j] for j,e in enumerate(EXPERTS)},**{f'th_{e}':ths[e] for e in EXPERTS},'runtime_s':time.perf_counter()-t0})
 per=pd.DataFrame(rows); per.to_csv(out/'per_sensor_test_metrics_v7.csv',index=False);pd.DataFrame(exrows).to_csv(out/'expert_test_metrics_v7.csv',index=False);pd.DataFrame(cfg).to_csv(out/'coordinator_config_v7.csv',index=False)
 # micro aggregate confusion + macro auc/mcc reporting
 agg=[]
 for m,g in per.groupby('method'):
  tp,tn,fp,fn=[int(g[x].sum()) for x in ['tp','tn','fp','fn']]; prec=tp/(tp+fp) if tp+fp else 0;rec=tp/(tp+fn) if tp+fn else 0;f1=2*prec*rec/(prec+rec) if prec+rec else 0;fpr=fp/(fp+tn) if fp+tn else 0;spec=tn/(tn+fp) if tn+fp else 0
  agg.append({'method':m,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'accuracy':(tp+tn)/(tp+tn+fp+fn),'precision':prec,'recall':rec,'f1':f1,'fpr':fpr,'specificity':spec,'balanced_accuracy':(rec+spec)/2,'mcc_macro':g.mcc.mean(),'roc_auc_macro':g.roc_auc.mean(),'pr_auc_macro':g.pr_auc.mean(),'reward':.45*f1+.30*rec+.15*prec-.10*fpr})
 pd.DataFrame(agg).sort_values('f1',ascending=False).to_csv(out/'overall_test_metrics_v7.csv',index=False)
 print(pd.DataFrame(agg).sort_values('f1',ascending=False).to_string(index=False))
if __name__=='__main__':main()
