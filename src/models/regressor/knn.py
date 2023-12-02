import pandas as pd
import numpy as np
from models.base import BaseModel
from dataclasses import dataclass
from typing import Dict, List
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

from const import Const


@dataclass
class KNNRegressor(BaseModel):
    model: KNeighborsRegressor
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
        
        return self.model.fit(X=x, y=y)
    
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
        
        return None
    
    def optimize_hyper_params(
        self, 
        df: pd.DataFrame,
        features: List[str],
        targets: List[str], 
        **hparams: Dict,
    ):
        config = {
            "n_neighbors": hparams.get("n_neighbors", self.config.n_neighbors),
            "weights": hparams.get("weights", self.config.weights),
            "algorithm": hparams.get("algorithm", self.config.algorithm),
            "leaf_size": hparams.get("leaf_size", self.config.leaf_size),
            "p": hparams.get("p", self.config.p),
            "metric": hparams.get("metric", self.config.metric),
            "metric_params": hparams.get("metric_params", self.config.metric_params),
            "n_jobs": hparams.get("n_jobs", self.config.n_jobs),
        }
        
        model = KNeighborsRegressor(**config)
        
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
            
            if self.config.use_predict_proba:
                y_pred = self.model.predict_proba(X=x_valid)
            else:
                y_pred = self.model.predict(X=x_valid)
            accuaraies.append(metrics.accuracy_score(y_valid, y_pred))

        return -1.0 * np.mean(accuaraies)