import pandas as pd
import numpy as np
from const import Const
from models.base import BaseModel
from dataclasses import dataclass
from typing import Dict, List
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder


@dataclass
class SVMClassifier(BaseModel):
    model: SVC
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
        
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes is None")
        
        oh = OneHotEncoder(sparse=False).fit([[i] for i in range(num_classes)])
        pred = self.model.predict(X=x.values)
        return oh.transform(pred.reshape(-1, 1))
    
    def predict_proba(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict_proba(X=x)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")

        assert len("linear") == len(self.model.kernel), "kernel must be linear"
                
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
            "C": hparams.get("C", self.config.C),
            "kernel": hparams.get("kernel", self.config.kernel),
            "degree": hparams.get("degree", self.config.degree),
            "gamma": hparams.get("gamma", self.config.gamma),
            "coef0": hparams.get("coef0", self.config.coef0),
            "shrinking": hparams.get("shrinking", self.config.shrinking),
            "tol": hparams.get("tol", self.config.tol),
            "cache_size": hparams.get("cache_size", self.config.cache_size),
            "epsilon": hparams.get("epsilon", self.config.epsilon),
            "verbose": hparams.get("verbose", self.config.verbose),
            "max_iter": hparams.get("max_iter", self.config.max_iter),
        }
        
        model = SVC(**config)
        
        accuaraies = []
        for fold in range(self.config.num_folds):
            x_train, y_train = df[df[Const.FOLD_ID]!=fold][features], df[df[Const.FOLD_ID]!=fold][targets]
            x_valid, y_valid = df[df[Const.FOLD_ID]!=fold][features], df[df[Const.FOLD_ID]!=fold][targets]

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
