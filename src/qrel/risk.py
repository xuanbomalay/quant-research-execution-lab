import numpy as np
import pandas as pd
from scipy.optimize import minimize
ASSETS=['AAPL', 'MSFT', 'IBM', 'INTC', 'KO', 'PEP', 'MCD', 'PG', 'XOM', 'CVX', 'CAT', 'MMM']
GROUP6={'consumer': [4, 5, 6, 7], 'energy': [8, 9], 'industrial': [10, 11], 'tech': [0, 1, 2, 3]}
FALLBACK6=[]

def caps_ok(w,budget=.8,tol=1e-7):
    return abs(np.sum(w)-budget)<tol and np.min(w)>=-tol and np.max(w)<=.15+tol and all(np.sum(w[j])<=.35+tol for j in GROUP6.values())

def feasible_budget(v,budget=.8):
    if budget==0:return np.zeros(len(ASSETS))
    v=np.asarray(v,float); v=np.nan_to_num(v,nan=0,posinf=0,neginf=0)
    objective=lambda w:np.sum((w-v)**2)
    con=[{'type':'eq','fun':lambda w:w.sum()-budget}]+[{'type':'ineq','fun':lambda w,j=j:.35-w[j].sum()} for j in GROUP6.values()]
    res=minimize(objective,np.ones(len(v))*budget/len(v),bounds=[(0,.15)]*len(v),constraints=con,method='SLSQP',options={'ftol':1e-12,'maxiter':150})
    if not res.success or not caps_ok(res.x,budget):
        FALLBACK6.append({'where':'projection','reason':res.message,'budget':budget}); fallback=np.ones(len(v))*budget/len(v)
        assert caps_ok(fallback,budget); return fallback
    return res.x

def limited_targets(opens,desired,turnover_cap=.25,cost=.0005,initial=10000.):
    cash=initial;units=np.zeros(desired.shape[1]);rows=[];audits=[]
    def solve(w,old,E):
        post=E
        for _ in range(40):post=E-np.abs(w*post-old).sum()*cost
        return post,w*post-old
    for count,(d,target) in enumerate(desired.iterrows()):
        p=opens.loc[d].to_numpy();nxt=opens.shift(-1).loc[d].to_numpy();old=units*p;E=cash+old.sum();oldw=old/E
        # 若行情漂移使权重超过上限，先做最小比例缩减；超出换手限制的强制降险单独登记。
        repair=min(1.,.8/max(oldw.sum(),1e-12),.15/max(oldw.max(),1e-12),*[.35/max(oldw[j].sum(),1e-12) for j in GROUP6.values()])
        safe=oldw*repair;want=target.to_numpy(float);post,delta=solve(want,old,E);limited=False
        if count>0 and np.abs(delta).sum()/E>turnover_cap:
            low,high=0.,1.
            for _ in range(35):
                mid=(low+high)/2;candidate=safe+mid*(want-safe);p0,d0=solve(candidate,old,E)
                if np.abs(d0).sum()/E<=turnover_cap:low=mid
                else:high=mid
            want=safe+low*(want-safe);post,delta=solve(want,old,E);limited=True
        turn=np.abs(delta).sum()/E
        forced=bool(count>0 and turn>turnover_cap+1e-6)
        fees=np.abs(delta).sum()*cost;units+=delta/p;cash-=delta.sum()+fees
        rows.append(want.copy());audits.append({'date':d,'desired_budget':target.sum(),'executed_budget':want.sum(),'turnover':turn,'limited':limited,'forced_risk_override':forced,'startup':count==0})
        assert cash+units@nxt>0
    return pd.DataFrame(rows,index=desired.index,columns=desired.columns),pd.DataFrame(audits).set_index('date')

