import pandas as pd
from typing import List, Union, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import (
    # encoder
    LabelEncoder, 
    OneHotEncoder, 
    OrdinalEncoder,
    # scaler
    StandardScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
)

NUMERICAL_TYPES = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]

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

@dataclass
class Encoder:
    # one_hot_encoder = OneHotEncoder(sparse=False)
    # lable_encoder = LabelEncoder()
    # oridinal_encoder = OrdinalEncoder()
    encoder: Union[OneHotEncoder, LabelEncoder, OrdinalEncoder]
    
    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        columns: List[str],
    ):
        if self.encoder is None:
            raise ValueError("Encoder is None")
        
        for col in columns:
            if train_df is not None:
                train_df[col] = self.encoder.fit_transform(train_df[col])
                if test_df is not None:
                    test_df[col] = self.encoder.transform(test_df[col])

        return train_df, test_df
    
@dataclass
class Preprocessor:
    scaler: Scaler = None
    encoder: Encoder = None
    EPS = 1e-8
    
    # Todo:
    # 1. Remove Outliers
    # 2. Remove Target Imbalance
    
    def fit_scaling(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        columns: List[str],
    ):
        if self.scaler is None:
            return train_df, test_df
        
        for col in columns:
            if train_df is not None:
                train_df[col] = self.scaler.fit_transform(train_df[col])
                if test_df is not None:
                    test_df[col] = self.scaler.transform(test_df[col])
                    
        return train_df, test_df
    
    def fit_encoding(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        columns: List[str],
    ):
        return self.encoder.fit(train_df, test_df, columns)
    
    def feature_engineering(
        self,
        df: pd.DataFrame,
        features: Union[
            List[Tuple[str, str]],
            List[Tuple[int, int]],
            List[Tuple[float, float]],
        ],
        operator: str = None, # +, -, *, /, _,
    ) -> pd.DataFrame:
        for col1, col2 in features:
            df_col1 = df[col1]
            df_col2 = df[col2]
            if df_col1.dtype in NUMERICAL_TYPES and df_col2.dtype in NUMERICAL_TYPES:
                if operator is None or operator in ["_", "+"]:
                    df[f"{col1}+{col2}"] = df[col1] + df[col2]
                elif operator == "-":
                    df[f"{col1}-{col2}"] = df[col1] - df[col2]
                elif operator == "*":
                    df[f"{col1}*{col2}"] = df[col1] * df[col2]
                elif operator == "/":
                    df[f"{col1}/{col2}"] = df[col1] / (df[col2]+self.EPS)
                else:
                    df[f"{col1}+{col2}"] = df[col1] + df[col2]
                
            elif df_col1.dtype == "str" and df_col2.dtype == "str":
                if operator is None:
                    operator = "_"
                df[f"{col1}{operator}{col2}"] = df[col1] + operator + df[col2]
        return df
        
    def fill_missing_value(
        self, 
        df: pd.DataFrame, 
        fill_value,
        columns: List[str],
    ):
        return df.fillna({col: fill_value for col in columns})
    
    def change_data_type(
        self, 
        df: pd.DataFrame,
        dtype,
        columns: List[str],
    ):
        return df.astype({col: dtype for col in columns})