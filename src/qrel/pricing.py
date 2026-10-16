import numpy as np
from scipy.stats import norm

def bsm9(S,K,r,q,sigma,T,kind='call'):
    S=np.asarray(S,float);T=float(T)
    if T<=0:
        call=np.maximum(S-K,0);price=call if kind=='call' else np.maximum(K-S,0)
        delta=np.where(S>K,1.,np.where(S<K,0.,.5))-(kind=='put')
        return {'price':price,'delta':delta,'gamma':np.zeros_like(S),'vega':np.zeros_like(S),'theta':np.zeros_like(S)}
    if sigma<=0 or np.any(S<=0) or K<=0:raise ValueError('positive S,K,sigma required before expiry')
    root=np.sqrt(T);d1=(np.log(S/K)+(r-q+.5*sigma*sigma)*T)/(sigma*root);d2=d1-sigma*root;eq=np.exp(-q*T);er=np.exp(-r*T);phi=norm.pdf(d1)
    if kind=='call':
        price=S*eq*norm.cdf(d1)-K*er*norm.cdf(d2);delta=eq*norm.cdf(d1)
        theta=-S*eq*phi*sigma/(2*root)+q*S*eq*norm.cdf(d1)-r*K*er*norm.cdf(d2)
    elif kind=='put':
        price=K*er*norm.cdf(-d2)-S*eq*norm.cdf(-d1);delta=eq*(norm.cdf(d1)-1)
        theta=-S*eq*phi*sigma/(2*root)-q*S*eq*norm.cdf(-d1)+r*K*er*norm.cdf(-d2)
    else:raise ValueError('call or put')
    return {'price':price,'delta':delta,'gamma':eq*phi/(S*sigma*root),'vega':S*eq*phi*root,'theta':theta}

