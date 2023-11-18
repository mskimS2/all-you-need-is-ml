import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict
from sklearn.svm import SVC, SVR


@dataclass
class SVMClassifier(BaseModel):
    model: SVC
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.C=self.config.C
        self.model.kernel=self.config.kernel
        self.model.degree=self.config.degree
        self.model.gamma=self.config.gamma
        self.model.coef0=self.config.coef0
        self.model.shrinking=self.config.shrinking
        self.model.probability=self.config.probability
        self.model.tol=self.config.tol
        self.model.cache_size=self.config.cache_size
        self.model.class_weight=self.config.class_weight
        self.model.verbose=self.config.verbose
        self.model.max_iter=self.config.max_iter
        self.model.decision_function_shape=self.config.decision_function_shape
    
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
        
@dataclass
class SVMRegressor(BaseModel):
    model: SVR
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.C=self.config.C
        self.model.kernel=self.config.kernel
        self.model.degree=self.config.degree
        self.model.gamma=self.config.gamma
        self.model.coef0=self.config.coef0
        self.model.shrinking=self.config.shrinking
        self.model.tol=self.config.tol
        self.model.cache_size=self.config.cache_size
        self.model.epsilon=self.config.epsilon
        self.model.verbose=self.config.verbose
        self.model.max_iter=self.config.max_iter
    
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