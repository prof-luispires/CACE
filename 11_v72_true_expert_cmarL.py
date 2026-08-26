#!/usr/bin/env python3
import argparse, importlib.util
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v7',HERE/'09_confidence_coordinator_v72.py');v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
spec=importlib.util.spec_from_file_location('base',HERE/'baseline_actions_v4.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
EXPERTS=v7.EXPERTS

def met(*a): return v7.met(*a)
def best_th(s,y,idx,pen=.2):
 best=(.5,-9)
 for t in np.linspace(.05,.95,37):
  m=met(y[idx],s[idx]>=t); z=m['reward']-pen*m['fpr']
  if z>best[1]:best=(t,z)
 return float(best[0])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=42);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 rows=[];ers=[];cfg=[]
 for fi,f in enumerate(sorted(Path(a.data).glob('sensor_*_multivar_labeled.csv'))):
  df=pd.read_csv(f);df.columns=[c.lower() for c in df.columns];L=v7.labels(df);n=len(df);ntr=int(.5*n);nv=int(.2*n);tr=np.arange(ntr);va=np.arange(ntr,ntr+nv);te=np.arange(ntr+nv,n)
  C=[];EP=[]
  for e in EXPERTS:
   acts=base.EXPERT_ACTIONS[e]; P=np.column_stack([base.ACTION_LIBRARY[x](df) for x in acts]).astype(float)
   rr=np.array([met(L[e][tr],P[tr,j])['reward']-.25*met(L[e][tr],P[tr,j])['fpr'] for j in range(P.shape[1])]); w=np.exp(5*(rr-rr.max()));w=w/w.sum(); sc=P@w; th=best_th(sc,L[e],tr,.3); pred=(sc>=th).astype(int);C.append(sc);EP.append(pred);ers.append({'sensor':f.stem,'expert':e,'threshold':th,'best_action':acts[int(np.argmax(rr))],**met(L[e][te],pred[te],sc[te])})
  C=np.column_stack(C);EP=np.column_stack(EP)
  # generic single RL: best one action on train for global label
  acts=base.SINGLE_ACTIONS; gp=np.column_stack([base.ACTION_LIBRARY[x](df) for x in acts]); rr=[met(L['global'][tr],gp[tr,j])['reward'] for j in range(len(acts))];j=int(np.argmax(rr));single=gp[:,j]
  majority=(EP.sum(1)>=2).astype(int);orall=(EP.max(1)>0).astype(int)
  lr=LogisticRegression(max_iter=1000,class_weight='balanced',random_state=a.seed).fit(C[tr],L['global'][tr]);score=lr.predict_proba(C)[:,1];th=best_th(score,L['global'],va,.35);coord=(score>=th).astype(int)
  # gated confidence: coordinator plus high-confidence 2-expert agreement
  gate=((C>=.65).sum(1)>=2).astype(int);hyb=np.maximum(coord,gate)
  for name,p,sc in [('single_agent_rl_refactored',single,single),('or_all',orall,orall),('majority',majority,majority),('confidence_coordinator',coord,score),('confidence_hybrid',hyb,np.maximum(score,gate))]: rows.append({'sensor':f.stem,'method':name,**met(L['global'][te],p[te],sc[te])})
  cfg.append({'sensor':f.stem,'coord_threshold':th,'single_action':acts[j],**{f'coef_{e}':lr.coef_[0][k] for k,e in enumerate(EXPERTS)}})
 pd.DataFrame(rows).to_csv(out/'per_sensor_test_metrics_v72.csv',index=False);pd.DataFrame(ers).to_csv(out/'expert_test_metrics_v72.csv',index=False);pd.DataFrame(cfg).to_csv(out/'config_v72.csv',index=False)
 print(pd.DataFrame(rows).groupby('method')[['precision','recall','f1','fpr','balanced_accuracy','mcc']].mean().sort_values('f1',ascending=False))
if __name__=='__main__':main()
