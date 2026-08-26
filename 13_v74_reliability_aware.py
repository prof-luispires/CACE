#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v7',HERE/'09_confidence_coordinator_v72.py');v7=importlib.util.module_from_spec(spec);spec.loader.exec_module(v7)
spec=importlib.util.spec_from_file_location('base',HERE/'baseline_actions_v4.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
EXPERTS=v7.EXPERTS

def met(*a): return v7.met(*a)
def best_th(s,y,idx,pen=.35):
    best=(.5,-1e99)
    for t in np.linspace(.05,.95,37):
        m=met(y[idx],s[idx]>=t)
        z=m['reward']-pen*m['fpr']
        if z>best[1]: best=(float(t),z)
    return best[0]
def events(y):
    y=np.asarray(y,int); d=np.diff(np.r_[0,y,0]); return list(zip(np.where(d==1)[0],np.where(d==-1)[0]))
def event_metrics(y,p):
    ev=events(y); pev=events(p)
    if not ev: return dict(n_events=0,event_detection_rate=np.nan,event_precision=np.nan,mean_detection_delay=np.nan,median_detection_delay=np.nan,early_detection_rate=np.nan)
    hits=0; delays=[]; early=0
    for s,e in ev:
        q=np.where(np.asarray(p)[s:e]>0)[0]
        if len(q):
            hits+=1; delay=int(q[0]); delays.append(delay); early+=int(delay<=max(1,int(.2*(e-s))))
    phits=sum(any(max(s,a)<min(e,b) for s,e in ev) for a,b in pev)
    return dict(n_events=len(ev),event_detection_rate=hits/len(ev),event_precision=phits/len(pev) if pev else 0,
                mean_detection_delay=float(np.mean(delays)) if delays else np.nan,
                median_detection_delay=float(np.median(delays)) if delays else np.nan,
                early_detection_rate=early/len(ev))

def reliability_from_validation(y,p):
    """Fixed a-priori reliability mapping. No test-set information is used.
    Balances point-wise discrimination (MCC/F1), event coverage, and false-alarm control.
    """
    m=met(y,p); em=event_metrics(y,p)
    mcc=max(0.0,float(m['mcc']))
    f1=float(m['f1'])
    edr=0.0 if not np.isfinite(em['event_detection_rate']) else float(em['event_detection_rate'])
    specificity=1.0-float(m['fpr'])
    r=.35*mcc + .25*f1 + .20*edr + .20*specificity
    return float(np.clip(r,.05,1.0)), m, em

def fuse(C,EP,L,tr,va,te):
    # Reliability is estimated only from the validation subset after expert thresholds are frozen from training.
    rel=[]
    val_details=[]
    for j,e in enumerate(EXPERTS):
        r,m,em=reliability_from_validation(L[e][va],EP[va,j])
        rel.append(r); val_details.append((m,em))
    rel=np.asarray(rel,float)
    rw=rel/max(rel.sum(),1e-12)
    score=(C*rw).sum(1)
    th=best_th(score,L['global'],va,.35)
    coord=(score>=th).astype(int)
    # Reliability-weighted expert agreement; avoids a fixed number-of-agents rule.
    active=(EP>0).astype(float)
    vote=(active*rw).sum(1)
    # vote threshold is selected on validation, not test
    vote_th=best_th(vote,L['global'],va,.45)
    gate=(vote>=vote_th).astype(int)
    # Conservative hybrid: require either calibrated score or reliability-weighted agreement.
    hybrid=np.maximum(coord,gate)
    return score,coord,hybrid,rel,rw,th,vote,vote_th,val_details

def run_folder(data,seed,scenario):
    rows=[]; ers=[]; abl=[]; cfg=[]
    for f in sorted(Path(data).glob('sensor_*_multivar_labeled.csv')):
        df=pd.read_csv(f); df.columns=[c.lower() for c in df.columns]; L=v7.labels(df)
        n=len(df); ntr=int(.5*n); nv=int(.2*n); tr=np.arange(ntr); va=np.arange(ntr,ntr+nv); te=np.arange(ntr+nv,n)
        C=[]; EP=[]; ex_cfg={}
        for e in EXPERTS:
            acts=base.EXPERT_ACTIONS[e]; P=np.column_stack([base.ACTION_LIBRARY[x](df) for x in acts]).astype(float)
            rr=np.array([met(L[e][tr],P[tr,j])['reward']-.25*met(L[e][tr],P[tr,j])['fpr'] for j in range(P.shape[1])])
            w=np.exp(5*(rr-rr.max())); w=w/w.sum(); sc=P@w; th=best_th(sc,L[e],tr,.3); pred=(sc>=th).astype(int)
            C.append(sc); EP.append(pred)
            ex_cfg[e]=(th,acts[int(np.argmax(rr))])
            ers.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'expert':e,'threshold':th,'best_action':acts[int(np.argmax(rr))],**met(L[e][te],pred[te],sc[te]),**event_metrics(L[e][te],pred[te])})
        C=np.column_stack(C); EP=np.column_stack(EP)
        acts=base.SINGLE_ACTIONS; gp=np.column_stack([base.ACTION_LIBRARY[x](df) for x in acts]); rr=[met(L['global'][tr],gp[tr,j])['reward'] for j in range(len(acts))]; j=int(np.argmax(rr)); single=gp[:,j]
        majority=(EP.sum(1)>=2).astype(int); orall=(EP.max(1)>0).astype(int)
        score,coord,hyb,rel,rw,cth,vote,vth,vd=fuse(C,EP,L,tr,va,te)
        methods=[('single_agent_rl_refactored',single,single),('or_all',orall,orall),('majority',majority,majority),('reliability_coordinator',coord,score),('reliability_hybrid',hyb,np.maximum(score,vote))]
        for name,p,sc in methods:
            rows.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'method':name,**met(L['global'][te],p[te],sc[te]),**event_metrics(L['global'][te],p[te])})
        abl.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'ablation':'full',**met(L['global'][te],hyb[te])})
        # Strict ablation: no test re-tuning; recompute reliability/thresholds from the same validation split using remaining experts only.
        for k,e in enumerate(EXPERTS):
            cols=[q for q in range(4) if q!=k]
            subE=[EXPERTS[q] for q in cols]
            r=[]
            for q,ename in zip(cols,subE):
                rv,_,_=reliability_from_validation(L[ename][va],EP[va,q]); r.append(rv)
            r=np.asarray(r,float); rw2=r/max(r.sum(),1e-12)
            s2=(C[:,cols]*rw2).sum(1); t2=best_th(s2,L['global'],va,.35); p2=(s2>=t2).astype(int)
            v2=(EP[:,cols]*rw2).sum(1); vt2=best_th(v2,L['global'],va,.45); g2=(v2>=vt2).astype(int); h2=np.maximum(p2,g2)
            abl.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'ablation':f'without_{e}',**met(L['global'][te],h2[te])})
        cfg.append({'scenario':scenario,'seed':seed,'sensor':f.stem,'single_action':acts[j],'coord_threshold':cth,'vote_threshold':vth,
                    **{f'reliability_{e}':rel[q] for q,e in enumerate(EXPERTS)},**{f'weight_{e}':rw[q] for q,e in enumerate(EXPERTS)},
                    **{f'expert_threshold_{e}':ex_cfg[e][0] for e in EXPERTS}})
    return rows,ers,abl,cfg

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--out',required=True); ap.add_argument('--scenarios',default='low,medium,high'); ap.add_argument('--seeds',default='42,123,321,777,999'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); R=[];E=[];A=[];C=[]
    for scen in a.scenarios.split(','):
        for seed in [int(x) for x in a.seeds.split(',')]:
            r,e,ab,c=run_folder(Path(a.data_root)/scen/f'seed_{seed}',seed,scen); R+=r;E+=e;A+=ab;C+=c
    d=pd.DataFrame(R); ed=pd.DataFrame(E); ad=pd.DataFrame(A); cd=pd.DataFrame(C)
    d.to_csv(out/'overall_long_v74.csv',index=False); ed.to_csv(out/'expert_event_metrics_v74.csv',index=False); ad.to_csv(out/'ablation_long_v74.csv',index=False); cd.to_csv(out/'config_v74.csv',index=False)
    metrics=['precision','recall','f1','fpr','balanced_accuracy','mcc','event_detection_rate','mean_detection_delay','early_detection_rate']
    s=d.groupby(['scenario','method'])[metrics].agg(['mean','std']).reset_index(); s.columns=['_'.join(x).strip('_') for x in s.columns]; s.to_csv(out/'scenario_summary_v74.csv',index=False)
    tests=[]
    for scen,g in d.groupby('scenario'):
        piv=g.pivot_table(index=['seed','sensor'],columns='method',values='f1')
        for b in ['single_agent_rl_refactored','majority','or_all']:
            x=piv['reliability_hybrid']; y=piv[b]; stat,p=wilcoxon(x,y,zero_method='wilcox',alternative='two-sided')
            tests.append({'scenario':scen,'comparison':f'reliability_hybrid vs {b}','mean_diff_f1':float((x-y).mean()),'relative_gain_pct':float(100*(x.mean()-y.mean())/max(y.mean(),1e-12)),'wilcoxon_stat':stat,'p_value':p,'n_pairs':len(x)})
    pd.DataFrame(tests).to_csv(out/'statistical_tests_v74.csv',index=False)
    absum=ad.groupby(['scenario','ablation'])[['f1','precision','recall','fpr','mcc']].agg(['mean','std']).reset_index(); absum.columns=['_'.join(x).strip('_') for x in absum.columns]; absum.to_csv(out/'ablation_summary_v74.csv',index=False)
    rcols=[f'reliability_{e}' for e in EXPERTS]+[f'weight_{e}' for e in EXPERTS]
    rs=cd.groupby('scenario')[rcols].agg(['mean','std']).reset_index(); rs.columns=['_'.join(x).strip('_') for x in rs.columns]; rs.to_csv(out/'reliability_summary_v74.csv',index=False)
    print(s[['scenario','method','f1_mean','precision_mean','recall_mean','fpr_mean','mcc_mean']].to_string(index=False))
    print('\nTESTS\n',pd.DataFrame(tests).to_string(index=False))
    print('\nABLATION\n',absum[['scenario','ablation','f1_mean','fpr_mean','mcc_mean']].to_string(index=False))
    print('\nRELIABILITY\n',rs.to_string(index=False))
if __name__=='__main__': main()
