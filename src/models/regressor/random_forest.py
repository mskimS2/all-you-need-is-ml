import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Union, Dict, List
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

from const import Const
from models.base import BaseModel


@dataclass
class RandomForestRegressor(BaseModel):
    model: RandomForestRegressor
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self, *args, **kwargs):
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self.model, k, v)
            
        for k, v in vars(self.config).items():
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
        )
    
    def predict(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict(X=x)
    
    def predict_proba(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict_proba(X=x)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")

        assert len(kwargs["columns"]) == len(self.model.feature_importances_)
                
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("lime") is not None:
            # TODO: lime value
            pass
        
        return pd.DataFrame.from_dict(
            {c: [v] for c, v in zip(kwargs["columns"], self.model.feature_importances_)}, 
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
            "criterion": hparams.get("criterion", self.config.max_features),
            "n_estimators": hparams.get("n_estimators", self.config.max_features),
            "max_depth": hparams.get("max_depth", self.config.max_features),
            "max_features": hparams.get("max_features", self.config.max_features),
            "max_depth": hparams.get("max_depth", self.config.max_depth),
            "min_samples_split": hparams.get("min_samples_split", self.config.min_samples_split),
            "min_samples_leaf": hparams.get("min_samples_leaf", self.config.min_samples_leaf),
            "min_weight_fraction_leaf": hparams.get("min_weight_fraction_leaf", self.config.min_weight_fraction_leaf),
            "max_features": hparams.get("max_features", self.config.max_features),
            "max_leaf_nodes": hparams.get("max_leaf_nodes", self.config.max_leaf_nodes),
            "min_impurity_decrease": hparams.get("min_impurity_decrease", self.config.min_impurity_decrease),
            "bootstrap": hparams.get("bootstrap", self.config.bootstrap),
            "oob_score": hparams.get("oob_score", self.config.oob_score),
            "n_jobs": hparams.get("n_jobs", self.config.n_jobs),
            "random_state": hparams.get("random_state", self.config.random_state),
            "verbose": hparams.get("verbose", self.config.verbose),
            "warm_start": hparams.get("warm_start", self.config.warm_start),
            "ccp_alpha": hparams.get("ccp_alpha", self.config.ccp_alpha),
            "max_samples": hparams.get("max_samples", self.config.max_samples),
        }
        
        model = RandomForestRegressor(**config)
        
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
