import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class RandomForest(BaseModel):
    model: Union[RandomForestClassifier, RandomForestRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.n_estimation=self.config.n_estimation
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
        self.model.ccp_alpha=self.config.ccp_alpha
        self.model.max_samples=self.config.max_samples
    
    def fit(self, *args, **kwargs):
        return self.model.fit(
            X=kwargs["X"],
            y=kwargs["y"],
            sample_weight=kwargs.get("sample_weight"),
        )
    
    def predict(self, *args, **kwargs):
        return self.model.predict(
            X=kwargs["X"],
        )
    
    def predict_proba(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")
                
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("lime") is not None:
            # TODO: lime value
            pass
        
        assert len(kwargs["columns"]) == len(self.model.feature_importances_)
        
        return pd.DataFrame.from_dict(
            {c: [v] for c, v in zip(kwargs["columns"], self.model.feature_importances_)}, 
            columns=["feature_importance"],
            orient="index",
        )