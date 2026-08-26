#!/usr/bin/env python3
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon
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
def events(y):
 y=np.asarray(y,int); d=np.diff(np.r_[0,y,0]); return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
def event_metrics(y,p):
 ev=events(y); pev=events(p)
 if not ev:return dict(n_events=0,event_detection_rate=np.nan,event_precision=np.nan,mean_detection_delay=np.nan,median_detection_delay=np.nan,early_detection_rate=np.nan)
 hits=0; delays=[]; early=0
 for s,e in ev:
  q=np.where(np.asarray(p)[s:e]>0)[0]
  if len(q):
   hits+=1; delay=int(q[0]);delays.append(delay); early+=int(delay<=max(1,int(.2*(e-s))))
 phits=sum(any(max(s,a)<min(e,b) for s,e in ev) for a,b in pev)
 return dict(n_events=len(ev),event_detection_rate=hits/len(ev),event_precision=phits/len(pev) if pev else 0,mean_detection_delay=float(np.mean(delays)) if delays else np.nan,median_detection_delay=float(np.median(delays)) if delays else np.nan,early_detection_rate=early/len(ev))
def fit_coord(C,y,tr,va,seed,cols=None):
 if cols is None: cols=list(range(C.shape[1]))
 X=C[:,cols]; lr=LogisticRegression(max_iter=1000,class_weight='balanced',random_state=seed).fit(X[tr],y[tr]);score=lr.predict_proba(X)[:,1];th=best_th(score,y,va,.35);pred=(score>=th).astype(int);return score,pred,th,lr

def run_folder(data,out,seed,scenario):
 rows=[];ers=[];evrows=[];abl=[];cfg=[]
 for f in sorted(Path(data).glob('sensor_*_multivar_labeled.csv')):
  df=pd.read_csv(f);df.columns=[c.lower() for c in df.columns];L=v7.labels(df);n=len(df);ntr=int(.5*n);nv=int(.2*n);tr=np.arange(ntr);va=np.arange(ntr,ntr+nv);te=np.arange(ntr+nv,n)
  C=[];EP=[]
  for e in EXPERTS:
   acts=base.EXPERT_ACTIONS[e];P=np.column_stack([base.ACTION_LIBRARY[x](df) for x in acts]).astype(float)
   rr=np.array([met(L[e][tr],P[tr,j])['reward']-.25*met(L[e][tr],P[tr,j])['fpr'] for j in range(P.shape[1])]);w=np.exp(5*(rr-rr.max()));w=w/w.sum();sc=P@w;th=best_th(sc,L[e],tr,.3);pred=(sc>=th).astype(int);C.append(sc);EP.append(pred)
   em=event_metrics(L[e][te],pred[te]);ers.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'expert':e,'threshold':th,'best_action':acts[int(np.argmax(rr))],**met(L[e][te],pred[te],sc[te]),**em})
  C=np.column_stack(C);EP=np.column_stack(EP)
  acts=base.SINGLE_ACTIONS;gp=np.column_stack([base.ACTION_LIBRARY[x](df) for x in acts]);rr=[met(L['global'][tr],gp[tr,j])['reward'] for j in range(len(acts))];j=int(np.argmax(rr));single=gp[:,j]
  majority=(EP.sum(1)>=2).astype(int);orall=(EP.max(1)>0).astype(int)
  score,coord,th,lr=fit_coord(C,L['global'],tr,va,seed);gate=((C>=.65).sum(1)>=2).astype(int);hyb=np.maximum(coord,gate)
  methods=[('single_agent_rl_refactored',single,single),('or_all',orall,orall),('majority',majority,majority),('confidence_coordinator',coord,score),('confidence_hybrid',hyb,np.maximum(score,gate))]
  for name,p,sc in methods: rows.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'method':name,**met(L['global'][te],p[te],sc[te]),**event_metrics(L['global'][te],p[te])})
  # ablation: refit coordinator after removing each expert; gate requires >=2 of remaining experts
  abl.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'ablation':'full',**met(L['global'][te],hyb[te])})
  for k,e in enumerate(EXPERTS):
   cols=[q for q in range(4) if q!=k];s2,p2,t2,l2=fit_coord(C,L['global'],tr,va,seed,cols);g2=((C[:,cols]>=.65).sum(1)>=2).astype(int);h2=np.maximum(p2,g2);abl.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'ablation':f'without_{e}',**met(L['global'][te],h2[te])})
  cfg.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'coord_threshold':th,'single_action':acts[j],**{f'coef_{e}':lr.coef_[0][k] for k,e in enumerate(EXPERTS)}})
 return rows,ers,abl,cfg

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--scenarios',default='low,medium,high');ap.add_argument('--seeds',default='42,123,321,777,999');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 R=[];E=[];A=[];C=[]
 for scen in a.scenarios.split(','):
  for seed in [int(x) for x in a.seeds.split(',')]:
   folder=Path(a.data_root)/scen/f'seed_{seed}';r,e,ab,c=run_folder(folder,out,seed,scen);R+=r;E+=e;A+=ab;C+=c
 pd.DataFrame(R).to_csv(out/'overall_long_v73.csv',index=False);pd.DataFrame(E).to_csv(out/'expert_event_metrics_v73.csv',index=False);pd.DataFrame(A).to_csv(out/'ablation_long_v73.csv',index=False);pd.DataFrame(C).to_csv(out/'config_v73.csv',index=False)
 d=pd.DataFrame(R); summary=d.groupby(['scenario','method'])[['precision','recall','f1','fpr','balanced_accuracy','mcc','event_detection_rate','mean_detection_delay','early_detection_rate']].agg(['mean','std']).reset_index();summary.columns=['_'.join(x).strip('_') for x in summary.columns];summary.to_csv(out/'scenario_summary_v73.csv',index=False)
 # paired tests hybrid vs single, and hybrid vs majority
 tests=[]
 for scen,g in d.groupby('scenario'):
  piv=g.pivot_table(index=['seed','sensor'],columns='method',values='f1')
  for b in ['single_agent_rl_refactored','majority','or_all']:
   x=piv['confidence_hybrid'];y=piv[b];stat,p=wilcoxon(x,y,zero_method='wilcox',alternative='two-sided');tests.append({'scenario':scen,'comparison':f'confidence_hybrid vs {b}','mean_diff_f1':float((x-y).mean()),'relative_gain_pct':float(100*(x.mean()-y.mean())/max(y.mean(),1e-12)),'wilcoxon_stat':stat,'p_value':p,'n_pairs':len(x)})
 pd.DataFrame(tests).to_csv(out/'statistical_tests_v73.csv',index=False)
 ab=pd.DataFrame(A);absum=ab.groupby(['scenario','ablation'])[['f1','precision','recall','fpr','mcc']].agg(['mean','std']).reset_index();absum.columns=['_'.join(x).strip('_') for x in absum.columns];absum.to_csv(out/'ablation_summary_v73.csv',index=False)
 print(summary[['scenario','method','f1_mean','precision_mean','recall_mean','fpr_mean','mcc_mean']].to_string(index=False));print('\nTests');print(pd.DataFrame(tests).to_string(index=False))
if __name__=='__main__':main()
