import pandas as pd
import numpy as np
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict
from xgboost import XGBClassifier, XGBRegressor


@dataclass
class XGBoost(BaseModel):
    model: Union[XGBClassifier, XGBRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.early_stopping_rounds=self.config.early_stopping_rounds
        self.model.learning_rate=self.config.learning_rate
        self.model.gamma=self.config.gamma
        self.model.max_depth=self.config.max_depth
        self.model.max_child_weight=self.config.max_child_weight
        self.model.max_delta_step=self.config.max_delta_step
        self.model.subsample=self.config.subsample
        self.model.sampling_method=self.config.sampling_method
        self.model.colsample_bytree=self.config.colsample_bytree
        self.model.alpha=self.config.alpha
        self.model.tree_method=self.config.tree_method
        self.model.scale_pos_weight=self.config.scale_pos_weight
        self.model.grow_policy=self.config.grow_policy
        self.model.max_leaves=self.config.max_leaves
        self.model.random_state=self.config.random_seed
        self.model.reg_lambda=self.config.reg_lambda
        self.model.device=self.config.device
    
    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
    
    def predict_proba(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")
        
        assert len(kwargs["columns"]) == len(self.model.feature_importances_)
        
        return pd.DataFrame.from_dict(
            {c: [v] for c, v in zip(kwargs["columns"], self.model.feature_importances_)}, 
            columns=["feature_importance"],
            orient="index",
        )