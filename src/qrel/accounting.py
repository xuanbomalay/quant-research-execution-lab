import numpy as np
import pandas as pd
L2_CONFIG={'assets': ['IWM', 'SPY'], 'start': '2008-01-01', 'end': '2023-01-01', 'window': 252, 'refit_every': 21, 'entry_z': 2.0, 'exit_z': 0.5, 'stop_z': 4.0, 'max_holding': 40, 'gross_target': 0.8, 'commission_bps': 1.0, 'slippage_bps': 4.0, 'borrow_annual': 0.03, 'funding_annual': 0.05, 'short_collateral': 1.5, 'initial_cash': 10000.0, 'p_threshold': 0.05, 'adf_maxlag': 4, 'single_window': 20, 'max_gross': 2.0, 'seed': 42, 'borrow_status': 'assumed availability, not historical observations'}

def l2_simulate_multi(opens,target,commission=None,slippage=None,borrow_rate=None,
                      funding=None,available=None,initial=None,collateral=None):
    commission=L2_CONFIG["commission_bps"] if commission is None else commission
    slippage=L2_CONFIG["slippage_bps"] if slippage is None else slippage
    borrow_rate=L2_CONFIG["borrow_annual"] if borrow_rate is None else borrow_rate
    funding=L2_CONFIG["funding_annual"] if funding is None else funding
    initial=L2_CONFIG["initial_cash"] if initial is None else initial
    collateral=L2_CONFIG["short_collateral"] if collateral is None else collateral
    if initial<=0 or collateral<1: raise ValueError("本金/担保比例无效")
    if not opens.columns.equals(target.columns): raise ValueError("价格与权重资产列必须同名同序")
    if not opens.index.is_monotonic_increasing or opens.index.has_duplicates:
        raise ValueError("价格索引需递增且唯一")
    if not target.index.equals(opens.loc[target.index[0]:target.index[-1]].index):
        raise ValueError("权重必须覆盖连续的共同交易日，不能跳过资金费用")
    if commission<0 or slippage<0 or borrow_rate<0 or funding<0: raise ValueError("费率不得为负")
    if target.isna().any().any() or (target.abs().sum(axis=1)>L2_CONFIG["max_gross"]).any(): raise ValueError("目标缺失/总敞口超限")
    p=opens.reindex(target.index); nxt=opens.shift(-1).reindex(target.index)
    ends=pd.Series(opens.index,index=opens.index).shift(-1).reindex(target.index)
    if (not np.isfinite(p.to_numpy()).all() or not np.isfinite(nxt.to_numpy()).all()
        or (p<=0).any().any() or (nxt<=0).any().any()):
        raise ValueError("执行和估值价格必须有限且为正")
    if available is None: available=pd.DataFrame(True,index=target.index,columns=target.columns)
    available=available.reindex(index=target.index,columns=target.columns)
    if available.isna().any().any(): raise ValueError("借券可得性缺失，不能默认为可借")
    cash=float(initial); units=np.zeros(len(target.columns)); rows=[]; trades=[]
    rate=(commission+slippage)/10000
    for date,w in target.iterrows():
        price=p.loc[date].to_numpy(float); future=nxt.loc[date].to_numpy(float)
        weights=w.to_numpy(float); blocked=bool(((weights<0)&~available.loc[date].to_numpy(bool)).any())
        if blocked: weights=np.zeros_like(weights)  # 不可借则关闭整个组合，已有空头买回。
        before=cash+units@price
        if before<=0: raise RuntimeError("权益非正，停止策略，不继续产生无效权重")
        old_value=units*price; post=before
        # 同时求解扣费后权益与各腿目标；成本在账户内只扣一次。
        for _ in range(40):
            delta=weights*post-old_value
            new_post=before-np.abs(delta).sum()*rate
            if abs(new_post-post)<1e-9: post=new_post; break
            post=new_post
        else: raise RuntimeError("持仓/费用求解未收敛")
        delta=weights*post-old_value
        delta[np.abs(delta)<1e-8]=0
        commission_cash=np.abs(delta)*commission/10000
        slippage_cash=np.abs(delta)*slippage/10000
        fees=commission_cash+slippage_cash
        units=units+delta/price; cash=cash-delta.sum()-fees.sum()
        after=cash+units@price
        assert np.isclose(after,before-fees.sum(),atol=1e-6)
        short_value=np.maximum(-units*price,0)
        restricted=collateral*short_value.sum()
        days=(ends.loc[date]-date).days
        borrow=short_value*borrow_rate*days/365
        finance=max(restricted-cash,0)*funding*days/365
        cash-=borrow.sum()+finance
        market_pnl=units*(future-price)
        end=cash+units@future
        assert np.isclose(end-before,market_pnl.sum()-fees.sum()-borrow.sum()-finance,atol=1e-6)
        if end<=0: raise RuntimeError("组合发生权益耗尽，停止并保留失败原因")
        row={"session":date,"end_session":ends.loc[date],"equity_before":before,
            "equity_after_trade":after,"cash":cash,"cash_restricted":restricted,
            "cash_free":cash-restricted,"equity_end":end,"net_return":end/before-1,
            "trade_notional":np.abs(delta).sum(),"transaction_cost":fees.sum(),
            "borrow_cost":borrow.sum(),"funding_cost":finance,"turnover":np.abs(delta).sum()/before,
            "gross_exposure":np.abs(units*price).sum()/after,"net_exposure":(units*price).sum()/after,
            "blocked_by_borrow":blocked,"market_pnl":market_pnl.sum(),
            "net_pnl":end-before,"long_value":np.maximum(units*price,0).sum(),
            "short_value":short_value.sum(),"holding":bool(np.abs(units).sum()>1e-8)}
        for j,a in enumerate(target.columns):
            row[f"units_{a}"]=units[j]; row[f"borrow_{a}"]=borrow[j]
            row[f"market_pnl_{a}"]=market_pnl[j]; row[f"position_value_{a}"]=units[j]*price[j]
            row[f"commission_{a}"]=commission_cash[j]; row[f"slippage_{a}"]=slippage_cash[j]
            if abs(delta[j])>1e-8:
                trades.append({"session":date,"instrument":a,"delta_model_units":delta[j]/price[j],
                    "notional":abs(delta[j]),"commission":commission_cash[j],"slippage":slippage_cash[j],
                    "model_price":price[j],"action":"buy/cover" if delta[j]>0 else "sell/short"})
        assert np.isclose(end,cash+units@future,atol=1e-6)
        rows.append(row)
    ledger=pd.DataFrame(rows).set_index("session")
    ledger["wealth"]=ledger.equity_end/initial
    ledger["drawdown"]=ledger.wealth/ledger.wealth.cummax().clip(lower=1)-1
    return ledger,pd.DataFrame(trades)

