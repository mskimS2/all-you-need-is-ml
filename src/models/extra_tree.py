import numpy as np
import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict, List
from sklearn import metrics
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor

from const import Const


@dataclass
class ExtraTree(BaseModel):
    model: Union[ExtraTreeClassifier, ExtraTreeRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self, *args, **kwargs):
        if isinstance(self.model, ExtraTreeClassifier):
            self.model = ExtraTreeClassifier(*args, **kwargs)
        elif isinstance(self.model, ExtraTreeRegressor):
            self.model = ExtraTreeRegressor(*args, **kwargs)
    
    def fit(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        y = kwargs.get("y")
        if y is None:
            raise ValueError("y is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype(np.float32)
                    
        return self.model.fit(
            X=x.values,
            y=y,
            sample_weight=kwargs.get("sample_weight"),
            check_input=kwargs.get("check_input", True),
        )
    
    def predict(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype(np.float32)
            
        return self.model.predict(
            X=x.values,
            check_input=kwargs.get("check_input", True),
        )
    
    def predict_proba(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype(np.float32)
            
        return self.model.predict_proba(
            X=x.values,
            check_input=kwargs.get("check_input", True),
        )
    
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
            "n_estimation": hparams.get("n_estimation", self.config.n_estimation),
            "criterion": hparams.get("criterion", self.config.criterion),
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
            "class_weight": hparams.get("class_weight", self.config.class_weight),
            "ccp_alpha": hparams.get("ccp_alpha", self.config.ccp_alpha),
            "max_samples": hparams.get("max_samples", self.config.max_samples),
        }
        
        if isinstance(self.model, ExtraTreeClassifier):
            model = ExtraTreeClassifier(**config)
        elif isinstance(self.model, ExtraTreeRegressor):
            model = ExtraTreeRegressor(**config)
        
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