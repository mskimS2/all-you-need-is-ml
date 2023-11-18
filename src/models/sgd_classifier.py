import numpy as np
import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder

@dataclass
class SgdClassifier(BaseModel):
    model: SGDClassifier
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.loss=self.config.loss
        self.model.penalty=self.config.penalty
        self.model.alpha=self.config.alpha
        self.model.copy_X=self.config.copy_X
        self.model.l1_ratio=self.config.l1_ratio
        self.model.fit_intercept=self.config.fit_intercept
        self.model.max_iter=self.config.max_iter
        self.model.tol=self.config.tol
        self.model.shuffle=self.config.shuffle
        self.model.verbose=self.config.verbose
        self.model.epsilon=self.config.epsilon
        self.model.n_jobs=self.config.n_jobs
        self.model.random_state=self.config.random_state
        self.model.learning_rate=self.config.learning_rate
        self.model.eta0=self.config.eta0
        self.model.power_t=self.config.power_t
        self.model.early_stopping=self.config.early_stopping
        self.model.validation_fraction=self.config.validation_fraction
        self.model.n_iter_no_change=self.config.n_iter_no_change
        self.model.class_weight=self.config.class_weight
        self.model.warm_start=self.config.warm_start
        self.model.average=self.config.average
    
    def fit(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype("float32")
                    
        return self.model.fit(
            X=x.values,
            y=kwargs.get("y"),
            sample_weight=kwargs.get("sample_weight"),
            coef_init=kwargs.get("coef_init"), 
            intercept_init=kwargs.get("intercept_init"),
        )
    
    def predict(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype("float32")
        
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes is None")
        
        oh = OneHotEncoder(sparse=False).fit([[i] for i in range(num_classes)])
        pred = self.model.predict(X=x.values)
        return oh.transform(pred.reshape(-1, 1))
    
    def predict_proba(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype(np.float32)
        
        return self.model.predict_proba(X=x.values)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("lime") is not None:
            # TODO: lime value
            pass
        
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")
        
        return None