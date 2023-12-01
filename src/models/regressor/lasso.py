import pandas as pd
import numpy as np
from models.base import BaseModel
from dataclasses import dataclass
from typing import Dict, List
from sklearn import metrics
from sklearn.linear_model import Lasso

from const import Const


@dataclass
class Lasso(BaseModel):
    model: Lasso
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self, *args, **kwargs):
        for k, v in vars(self.config).items():
            setattr(self.model, k, v)
            
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self.model, k, v)
    
    def fit(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        y = kwargs.get("y")        
        if y is None:
            raise ValueError("y is None")
    
        return self.model.fit(
            X=x,
            y=y,
            sample_weight=kwargs.get("sample_weight"),
            check_input=kwargs.get("sample_weight", True),
        )
    
    def predict(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict(X=x)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")
                
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("lime") is not None:
            # TODO: lime value
            pass
        
        return pd.DataFrame.from_dict(
            {c: [v] for c, v in zip(kwargs["columns"], self.model.coef_)}, 
            columns=["feature_importance"],
            orient="index",
        )
        
    def optimize_hyper_params(
        self, 
        df: pd.DataFrame,
        features: List[str],
        targets: List[str], 
        **hparams: Dict,
    ):
        config = {
            "alpha": hparams.get("alpha", self.config.alpha),
            "fit_intercept": hparams.get("fit_intercept", self.config.fit_intercept),
            "precompute": hparams.get("precompute", self.config.precompute),
            "copy_X": hparams.get("copy_X", self.config.copy_X),
            "max_iter": hparams.get("max_iter", self.config.max_iter),
            "tol": hparams.get("tol", self.config.tol),
            "warm_start": hparams.get("warm_start", self.config.warm_start),
            "positive": hparams.get("positive", self.config.positive),
            "random_state": hparams.get("random_state", self.config.random_state),
            "selection": hparams.get("selection", self.config.selection),
        }
        
        if isinstance(model, Lasso):
            model = Lasso(**config)
        
        accuaraies = []
        for fold in range(self.config.num_folds):
            x_train = df[df[Const.FOLD_ID]!=fold][features]
            y_train = df[df[Const.FOLD_ID]!=fold][targets]
            x_valid = df[df[Const.FOLD_ID]!=fold][features]
            y_valid = df[df[Const.FOLD_ID]!=fold][targets]

            model.fit(
                X=x_train,
                y=y_train,
                eval_set=[(x_valid, y_valid)],
                **config,
            )
            
            y_pred = self.model.predict(X=x_valid)
            accuaraies.append(metrics.accuracy_score(y_valid, y_pred))

        return -1.0 * np.mean(accuaraies)