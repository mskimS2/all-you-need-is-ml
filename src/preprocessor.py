from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from typing import List, Union, Tuple
from dataclasses import dataclass
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import (
    LabelEncoder, 
    OneHotEncoder, 
    OrdinalEncoder,
    StandardScaler,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
)

import utils
from type import Problem, Task
from const import Const

NUMERICAL_TYPES = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]

@dataclass
class Scaler:
    
    def __init__(
        self,
        scaler_name: str,
        scaler_params: dict,
    ):
        self.scaler = self._get_scaler(scaler_name, scaler_params)
        
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
        
    def fit(self, X: np.ndarray):
        return self.scaler.fit(X)
    
    def transform(self, X: np.ndarray):
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray):
        return self.scaler.fit_transform(X)

class Encoder:
    
    def __init__(
        self,
        encoder_name: str,
        encoder_params: dict,
    ) -> None:
        self.encoder = self._get_encoder(encoder_name, encoder_params)
        
    def _get_encoder(self, encoder_name: str, encoder_params: dict):
        if encoder_name == "onehot":
            return OneHotEncoder(**encoder_params)
        elif encoder_name == "label":
            return LabelEncoder(**encoder_params)
        elif encoder_name == "ordinal":
            return OrdinalEncoder(**encoder_params)
        else:
            raise ValueError(f"Invalid encoder name: {encoder_name}")
    
    def fit(self, X: np.ndarray):
        return self.encoder.fit(X)
    
    def transform(self, X: np.ndarray):
        return self.encoder.transform(X)
    
    def fit_transform(self, X: np.ndarray):
        return self.encoder.fit_transform(X)
    
@dataclass
class Preprocessor:

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_columns: List[str],
        problem: Problem,
        n_splits: int,
        shuffle: bool,
        random_state: int,
        encoder: Union[OneHotEncoder, LabelEncoder, OrdinalEncoder],
        scaler_name: str = "standard",
        scaler_params: dict = {},
        encoder_name: str = "onehot",
        encoder_params: dict = {},
        reduce_memory_usage: bool = True,
    ) -> Tuple(pd.DataFrame, Pipeline):
        
        if reduce_memory_usage:
            if df is not None:
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df)
                df = utils.reduce_memory_usage(df)
                
        df = self.created_fold(
            df=df,
            problem=problem,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            target_columns=target_columns,
        )
        
        pipeline = Pipeline(
            [
                ("scaler", Scaler(scaler_name, scaler_params)),
                ("encoder", Encoder(encoder_name, encoder_params)),
            ] 
        )
        
        return df, pipeline
    
    def check_problem_type(
        self,
        df: pd.DataFrame,
        task: Task,
        target_columns: List[str],
    ) -> Tuple[Problem, int]:
        num_labels = len(np.unique(df[target_columns].values))
        if Task.type.get(task) is not None:
            task = Task.type.get(task)
            if task == Task.type[Const.CLASSIFICATION]:
                problem_type = Const.MULTI_LABEL_CLASSIFICATION
                if num_labels == 2:
                    problem_type = Const.BINARY_CLASSIFICATION
                    
            elif task == Task.type[Const.REGRESSION]:
                problem_type = Const.MULTI_COLUMN_REGRESSION
                if num_labels == 1:
                    problem_type = Const.SINGLE_COLUMN_REGRESSION
        else:
            raise Exception("Problem type not understood")

        return problem_type, num_labels
        
    def created_fold(
        self,
        df: pd.DataFrame,
        problem: Problem,
        n_splits: int,
        shuffle: bool,
        random_state: int,
        target_columns: List[str],
    ):
        if Const.FOLD_ID in df.columns:
            print("`fold_id` column already exists from input dataframe.")
            return df
        
        df[Const.FOLD_ID] = -1
        if problem in [Const.BINARY_CLASSIFICATION, Const.MULTI_CLASS_CLASSIFICATION]:
            y = df[target_columns].values.ravel()
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            for fold, (_, valid_indicies) in enumerate(skf.split(X=df, y=y)):
                df.loc[valid_indicies, Const.FOLD_ID] = fold
                
        elif problem in [Const.SINGLE_COLUMN_REGRESSION]:
            y = df[target_columns].values.ravel()
            num_bins = min(int(np.floor(1 + np.log2(len(df)))), 10)
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            df["bins"] = pd.cut(df[target_columns].values.ravel(), bins=num_bins, labels=False)
            for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=df.bins.values)):
                df.loc[valid_indicies, Const.FOLD_ID] = fold
            df = df.drop("bins", axis=1)
            
        elif problem in [Const.MULTI_COLUMN_REGRESSION]:
            # Todo: MultilabelStratifiedKFold
            y = df[target_columns].values.ravel()
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=y)):
                df.loc[valid_indicies, Const.FOLD_ID] = fold
        
        elif problem in [Const.MULTI_LABEL_CLASSIFICATION]:
            # Todo: MultilabelStratifiedKFold
            y = df[target_columns].values.ravel()
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=y)):
                df.loc[valid_indicies, Const.FOLD_ID] = fold

        else:
            raise Exception("Invalid problem type for created fold.")
            
        return df
    
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