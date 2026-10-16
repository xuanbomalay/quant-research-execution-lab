import numpy as np
GROUP6={'consumer': [4, 5, 6, 7], 'energy': [8, 9], 'industrial': [10, 11], 'tech': [0, 1, 2, 3]}

def step_account8(weight,wealth,peak,desired,ret,cost=.0005,penalty=.05):
    old=np.asarray(weight,float);desired=np.asarray(desired,float)
    repair=min(1.,.8/max(old.sum(),1e-12),.15/max(old.max(),1e-12),*[.35/max(old[j].sum(),1e-12) for j in GROUP6.values()])
    safe=old*repair
    def trade(w):
        post=1.
        for _ in range(20):post=1-cost*np.abs(w*post-old).sum()
        delta=w*post-old;return post,delta
    post,delta=trade(desired);turn=np.abs(delta).sum()
    if turn>.25:
        lo,hi=0.,1.
        for _ in range(28):
            mid=(lo+hi)/2;p,d=trade(safe+mid*(desired-safe))
            if np.abs(d).sum()<=.25:lo=mid
            else:hi=mid
        desired=safe+lo*(desired-safe);post,delta=trade(desired)
    fee=cost*np.abs(delta).sum();cash=1-(desired*post).sum()-fee
    end_values=desired*post*(1+ret);growth=cash+end_values.sum();new_wealth=wealth*growth;new_peak=max(peak,new_wealth)
    dd=1-new_wealth/new_peak;reward=100*np.log(max(growth,1e-12))-penalty*max(dd-.10,0)
    return end_values/growth,new_wealth,new_peak,reward,desired,fee,np.abs(delta).sum()

