import pandas as pd
from dataclasses import dataclass
from typing import List
from sklearn.preprocessing import (
    StandardScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
)

@dataclass
class Scaler:
    scaler_name: str
    scaler_params: dict
    features: List[str]
    
    def __post_init__(self):
        self.scaler = self._get_scaler(self.scaler_name, **self.scaler_params)
        
    def _get_scaler(self, scaler_name: str, scaler_params: dict):
        if scaler_name == "standard":
            return StandardScaler(**scaler_params)
        elif scaler_name == "max_abs":
            return MaxAbsScaler(**scaler_params)
        elif scaler_name == "min_max":
            return MinMaxScaler(**scaler_params)
        elif scaler_name == "robust":
            return RobustScaler(**scaler_params)
        else:
            raise ValueError(f"Invalid scaler name: {scaler_name}")
        
    def fit(self, df: pd.DataFrame):
        return self.scaler.fit(df[self.features])
    
    def transform(self, df: pd.DataFrame):
        return self.scaler.transform(df[self.features])
    
    def fit_transform(self, df: pd.DataFrame):
        return self.scaler.fit_transform(df[self.features])
        
    
    
    