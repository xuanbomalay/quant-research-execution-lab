import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

class TrainClip(BaseEstimator,TransformerMixin):
    def __init__(self,low=.01,high=.99): self.low=low; self.high=high
    def fit(self,X,y=None):
        arr=np.asarray(X,float); self.lo_=np.nanquantile(arr,self.low,axis=0); self.hi_=np.nanquantile(arr,self.high,axis=0); return self
    def transform(self,X): return np.clip(np.asarray(X,float),self.lo_,self.hi_)

def regression_pipe(cols,model=None,clip=False,indicator=True):
    numeric=[]
    if clip: numeric.append(('clip',TrainClip()))
    numeric.extend([('impute',SimpleImputer(strategy='median',add_indicator=indicator,keep_empty_features=True)),('scale',StandardScaler())])
    return Pipeline([('features',ColumnTransformer([('numeric',Pipeline(numeric),cols)])),('model',Ridge(alpha=10) if model is None else model)])

