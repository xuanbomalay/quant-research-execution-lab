import unittest
import numpy as np
import pandas as pd
from qrel.accounting import l2_simulate_multi
from qrel.pricing import bsm9
from qrel.preprocessing import TrainClip
from qrel.rl_core import step_account8

class ResearchCoreTests(unittest.TestCase):
    def test_manual_fee_self_finance(self):
        d=pd.to_datetime(['2024-01-05','2024-01-08']);p=pd.DataFrame({'A':[100.,100.],'B':[50.,50.]},index=d);w=pd.DataFrame([[.5,.3]],index=d[:1],columns=p.columns)
        x,_=l2_simulate_multi(p,w,commission=10,slippage=0,borrow_rate=0,funding=0,initial=1000)
        self.assertAlmostEqual(x.equity_end.iloc[0],1000/(1+.001*.8),places=7)
    def test_long_short_weekend_borrow(self):
        d=pd.to_datetime(['2024-01-05','2024-01-08']);p=pd.DataFrame({'A':[100.,100.],'B':[100.,100.]},index=d);w=pd.DataFrame([[.4,-.4]],index=d[:1],columns=p.columns)
        x,_=l2_simulate_multi(p,w,commission=0,slippage=0,borrow_rate=.03,funding=0,initial=1000)
        self.assertAlmostEqual(x.borrow_cost.iloc[0],400*.03*3/365,places=10)
    def test_unavailable_borrow(self):
        d=pd.date_range('2024-01-01',periods=3);p=pd.DataFrame({'A':[100.,102.,101.],'B':[50.,49.,48.]},index=d)
        w=pd.DataFrame([[.4,-.4],[.4,-.4]],index=d[:2],columns=p.columns);a=pd.DataFrame(True,index=w.index,columns=w.columns);a.iloc[1,1]=False
        x,_=l2_simulate_multi(p,w,commission=3,slippage=2,borrow_rate=.03,funding=.05,available=a,initial=1000)
        self.assertTrue(x.blocked_by_borrow.iloc[1]);self.assertLess(abs(x.gross_exposure.iloc[1]),1e-8)
        np.testing.assert_allclose(x.net_pnl,x.market_pnl-x.transaction_cost-x.borrow_cost-x.funding_cost,atol=1e-7)
    def test_common_window(self):
        d=pd.date_range('2024-01-01',periods=4);p=pd.DataFrame({'A':[100.,100.,100.,100.]},index=d);w=pd.DataFrame({'A':[.5,.5]},index=d[[0,2]])
        with self.assertRaises(ValueError):l2_simulate_multi(p,w)
    def test_negative_cost(self):
        d=pd.date_range('2024-01-01',periods=2);p=pd.DataFrame({'A':[100.,100.]},index=d);w=pd.DataFrame({'A':[.5]},index=d[:1])
        with self.assertRaises(ValueError):l2_simulate_multi(p,w,commission=-1)
    def test_put_call_parity(self):
        c=bsm9(100,100,.05,.02,.2,1,kind='call')['price'];p=bsm9(100,100,.05,.02,.2,1,kind='put')['price']
        self.assertAlmostEqual(float(c-p),100*np.exp(-.02)-100*np.exp(-.05),places=10)
    def test_greeks_finite_difference(self):
        h=.01;price=lambda s:float(bsm9(s,100,.05,.02,.2,1)['price']);g=bsm9(100,100,.05,.02,.2,1)
        self.assertAlmostEqual((price(100+h)-price(100-h))/(2*h),float(g['delta']),places=6)
        self.assertAlmostEqual((price(100+h)-2*price(100)+price(100-h))/h**2,float(g['gamma']),places=6)
    def test_train_only_clipping(self):
        train=np.array([[1.,10.],[2.,20.],[3.,30.],[4.,40.]]);c=TrainClip().fit(train);old=c.lo_.copy();out=c.transform(np.array([[1000.,-1000.]]))
        np.testing.assert_array_equal(c.lo_,old);self.assertLessEqual(out[0,0],c.hi_[0]);self.assertGreaterEqual(out[0,1],c.lo_[1])
    def test_rl_cost_once(self):
        w,v,p,r,t,fee,turn=step_account8(np.zeros(12),1.,1.,np.ones(12)*.2/12,np.zeros(12),penalty=0)
        self.assertAlmostEqual(v,1/(1+.0005*.2),places=9);self.assertAlmostEqual(r,100*np.log(v),places=9)
    def test_rl_constraints(self):
        w,v,p,r,t,fee,turn=step_account8(np.zeros(12),1.,1.,np.ones(12)*.8/12,np.ones(12)*.01)
        self.assertLessEqual(turn,.250001);self.assertLessEqual(t.sum(),.800001);self.assertLessEqual(t.max(),.150001)
        np.testing.assert_allclose(v,(1-fee)*(1+t.sum()*.01),atol=1e-8)
    def test_label_purge(self):
        end=pd.Series(pd.to_datetime(['2020-12-30','2021-01-04']));self.assertEqual((end<pd.Timestamp('2021-01-01')).tolist(),[True,False])
    def test_queue_not_touch(self):
        ahead=100;self.assertEqual(max(0,60-ahead),0);ahead=max(0,ahead-60);self.assertEqual(min(20,max(0,50-ahead)),10)

if __name__=='__main__':unittest.main()
