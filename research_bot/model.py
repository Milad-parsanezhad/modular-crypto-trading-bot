from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from .features import FEATURE_COLUMNS
from .regime import REGIME_COLUMNS


def make_dataset(df: pd.DataFrame, hurdle_bps: float = 0.0):
    x=df.copy(); x["future_return"]=x["close"].shift(-1)/x["close"]-1; hurdle=hurdle_bps/10000.0
    x["target_up"]=(x["future_return"]>hurdle).astype(int); cols=[c for c in FEATURE_COLUMNS+REGIME_COLUMNS if c in x.columns]
    return x.dropna(subset=["future_return"]), cols


def fit_predict_holdout(df: pd.DataFrame, train_fraction: float = 0.70, hurdle_bps: float = 0.0, random_state: int = 42):
    data,cols=make_dataset(df,hurdle_bps=hurdle_bps)
    if len(data)<300: raise ValueError("Need at least 300 usable rows for a research holdout.")
    split=int(len(data)*train_fraction); train=data.iloc[:split].copy(); test=data.iloc[split:].copy()
    model=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",HistGradientBoostingClassifier(learning_rate=0.05,max_iter=200,max_leaf_nodes=15,l2_regularization=0.5,random_state=random_state))])
    model.fit(train[cols],train["target_up"]); test["prob_up"]=model.predict_proba(test[cols])[:,1]
    return model,train,test,cols


def positions_from_probabilities(prob_up: pd.Series, upper: float = 0.56, lower: float = 0.44, allow_short: bool = False):
    p=pd.Series(prob_up)
    if allow_short: return pd.Series(np.where(p>upper,1.0,np.where(p<lower,-1.0,0.0)),index=p.index)
    return (p>upper).astype(float)
