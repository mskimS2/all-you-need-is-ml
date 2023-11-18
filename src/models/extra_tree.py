import numpy as np
import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor


@dataclass
class ExtraTree(BaseModel):
    model: Union[ExtraTreeClassifier, ExtraTreeRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.n_estimation=self.config.n_estimation
        self.model.criterion=self.config.criterion
        self.model.max_depth=self.config.max_depth
        self.model.min_samples_split=self.config.min_samples_split
        self.model.min_samples_leaf=self.config.min_samples_leaf
        self.model.min_weight_fraction_leaf=self.config.min_weight_fraction_leaf
        self.model.max_features=self.config.max_features
        self.model.max_leaf_nodes=self.config.max_leaf_nodes
        self.model.min_impurity_decrease=self.config.min_impurity_decrease
        self.model.bootstrap=self.config.bootstrap
        self.model.oob_score=self.config.oob_score
        self.model.n_jobs=self.config.n_jobs
        self.model.random_state=self.config.random_state
        self.model.verbose=self.config.verbose
        self.model.warm_start=self.config.warm_start
        self.model.class_weight=self.config.class_weight
        self.model.ccp_alpha=self.config.ccp_alpha
        self.model.max_samples=self.config.max_samples
    
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