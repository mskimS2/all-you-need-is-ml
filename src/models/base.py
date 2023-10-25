import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from sklearn.model_selection import KFold, StratifiedKFold

from logger import logger
from type import Problem, Task


def created_fold(
    df: pd.DataFrame,
    problem: Problem,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    target_columns: List[str],
):
    if "fold_id" in df.columns:
        logger.info("`fold_id` column already exists from input dataframe.")
        return df
    
    df["fold_id"] = -1
    if problem in (
        Problem.type.binary_classification, 
        Problem.type.multi_class_classification,
    ):
        y = df[target_columns].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (_, valid_indicies) in enumerate(skf.split(X=df, y=y)):
            df.loc[valid_indicies, "fold_id"] = fold
            
    elif problem in Problem.type.single_column_regression:
        y = df[target_columns].values
        num_bins = min(int(np.floor(1 + np.log2(len(df)))), 10)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        df["bins"] = pd.cut(df[target_columns].values.ravel(), bins=num_bins, labels=False)
        for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=df.bins.values)):
            df.loc[valid_indicies, "fold_id"] = fold
        df = df.drop("bins", axis=1)
        
    elif problem in Problem.type.multi_column_regression:
        # Todo: MultilabelStratifiedKFold
        y = df[target_columns].values
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=y)):
            df.loc[valid_indicies, "fold_id"] = fold
    
    elif problem in Problem.type.multi_label_classification:
        # Todo: MultilabelStratifiedKFold
        y = df[target_columns].values
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=y)):
            df.loc[valid_indicies, "fold_id"] = fold

    else:
        raise Exception("Invalid problem type for created fold.")
        
    return df

def check_problem_type(
    df: pd.DataFrame,
    task: Task,
    target_columns: List[str],
):
    num_labels = len(np.unique(df[target_columns].values))
    if task is not None:
        if task == Task.types.classification:
            problem_type = Problem.type.multi_label_classification
            if num_labels == 2:
                problem_type = Problem.type.binary_classification

        elif task == Task.types.regression:
            problem_type = Problem.type.multi_column_regression
            if num_labels == 1:
                problem_type = Problem.type.single_column_regression
                
        else:
            raise Exception("Problem type not understood")

    logger.info(f"problem type: {problem_type.name}, detected labels: {num_labels}")
    return problem_type

@dataclass
class Trainer:
    
    def fit(self, train_df: pd.DataFrame):
        raise NotImplementedError
    
    def predict(self, test_df: pd.DataFrame):
        raise NotImplementedError