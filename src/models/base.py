import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import KFold, StratifiedKFold

from logger import logger
from type import Problem


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