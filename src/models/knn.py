import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Dict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


@dataclass
class KNNClassifier(BaseModel):
    model: KNeighborsClassifier
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.n_neighbors=self.config.n_neighbors
        self.model.weights=self.config.weights
        self.model.algorithm=self.config.algorithm
        self.model.gamma=self.config.gamma
        self.model.leaf_size=self.config.leaf_size
        self.model.p=self.config.p
        self.model.metric=self.config.metric
        self.model.metric_params=self.config.metric_params
        self.model.n_jobs=self.config.n_jobs
    
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
    
    def predict_proba(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")

        return self.model.predict_proba(X=x)
    
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
        
@dataclass
class KNNRegressor(BaseModel):
    model: KNeighborsRegressor
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.n_neighbors=self.config.n_neighbors
        self.model.weights=self.config.weights
        self.model.algorithm=self.config.algorithm
        self.model.gamma=self.config.gamma
        self.model.leaf_size=self.config.leaf_size
        self.model.p=self.config.p
        self.model.metric=self.config.metric
        self.model.metric_params=self.config.metric_params
        self.model.n_jobs=self.config.n_jobs
    
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

        assert len("columns") == len(self.model.feature_importances_)
                
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("lime") is not None:
            # TODO: lime value
            pass
        
        return None