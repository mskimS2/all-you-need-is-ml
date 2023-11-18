import pandas as pd
import numpy as np
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict, List
from sklearn import metrics
from xgboost import XGBClassifier, XGBRegressor

from const import Const

@dataclass
class XGBoost(BaseModel):
    model: Union[XGBClassifier, XGBRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self, *args, **kwargs):
        if isinstance(self.model, XGBClassifier):
            self.model = XGBClassifier(*args, **kwargs)
        elif isinstance(self.model, XGBRegressor):
            self.model = XGBRegressor(*args, **kwargs)
    
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
            base_margin=kwargs.get("base_margin"),
            eval_set=kwargs.get("eval_set"),
            eval_metric=kwargs.get("eval_metric"),
            early_stopping_rounds=kwargs.get("early_stopping_rounds"),
            verbose=kwargs.get("verbose"),
            xgb_model=kwargs.get("xgb_model"),
            sample_weight_eval_set=kwargs.get("sample_weight_eval_set"),
            base_margin_eval_set=kwargs.get("base_margin_eval_set"),
            feature_weights=kwargs.get("feature_weights"),
            callbacks=kwargs.get("callbacks"),
        )
    
    def predict(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict(
            X=x,
            output_margin=kwargs.get("output_margin", False),
            validate_features=kwargs.get("validate_features", True),
            base_margin=kwargs.get("base_margin"),
            iteration_range=kwargs.get("iteration_range"),
        )
    
    def predict_proba(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict_proba(
            X=x,
            validate_features=kwargs.get("validate_features", True),
            base_margin=kwargs.get("base_margin"),
            iteration_range=kwargs.get("iteration_range"),
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
            "early_stopping_rounds": hparams.get("early_stopping_rounds", self.config.early_stopping_rounds),
            "learning_rate": hparams.get("learning_rate", self.config.learning_rate),
            "gamma": hparams.get("gamma", self.config.gamma),
            "max_depth": hparams.get("max_depth", self.config.max_depth),
            "max_child_weight": hparams.get("max_child_weight", self.config.max_child_weight),
            "max_delta_step": hparams.get("max_delta_step", self.config.max_delta_step),
            "subsample": hparams.get("subsample", self.config.subsample),
            "sampling_method": hparams.get("sampling_method", self.config.sampling_method),
            "colsample_bytree": hparams.get("colsample_bytree", self.config.colsample_bytree),
            "alpha": hparams.get("alpha", self.config.alpha),
            "tree_method": hparams.get("tree_method", self.config.tree_method),
            "scale_pos_weight": hparams.get("scale_pos_weight", self.config.scale_pos_weight),
            "grow_policy": hparams.get("grow_policy", self.config.grow_policy),
            "max_leaves": hparams.get("max_leaves", self.config.max_leaves),
            "random_seed": hparams.get("random_seed", self.config.random_seed),
            "reg_lambda": hparams.get("reg_lambda", self.config.reg_lambda),
            "device": hparams.get("device", self.config.device),
        }
        
        if isinstance(model, XGBClassifier):
            model = XGBClassifier(**config)
        elif isinstance(model, XGBRegressor):
            model = XGBRegressor(**config)
        
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