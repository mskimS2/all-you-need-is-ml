import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from sklearn.model_selection import KFold, StratifiedKFold

from models.base import BaseModel
from preprocessing import Encoder, Scaler, Preprocessor
from metrics import Metric
from logger import logger
from type import Problem, Task
from const import Const

@dataclass
class Trainer:
    model: BaseModel
    config: dict
    scaler: Scaler
    encoder: Encoder
    
    def __post_init__(self):
        self.preprocessor = Preprocessor(self.scaler, self.encoder)
    
    def fit(
        self, 
        train_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        features: List[str] = None,
        targets: List[str] = None,
    ):
        self.columns = features
        logger.info(f"configuration: {self.model.config}")
        metrics = Metric(self.config.problem_type)
        scores = []
        
        train_df = self.created_fold(
            train_df, 
            self.config.problem_type, 
            self.config.num_folds, 
            self.config.shuffle, 
            self.config.random_seed, 
            targets,
        )
        
        train_df, test_df = self.preprocessor.fit_scaling(train_df, test_df, targets)
        
        train_df, test_df = self.preprocessor.fit_encoding(train_df, test_df, targets)
        
        for fold in range(self.config.num_folds):
            x_train = train_df[train_df["fold_id"]!=fold][features]
            y_train = train_df[train_df["fold_id"]!=fold][targets]
            x_valid = train_df[train_df["fold_id"]!=fold][features]
            y_valid = train_df[train_df["fold_id"]!=fold][targets]

            ypred = []
            models = [self.model] * len(targets)
            for idx, _m in enumerate(models):
                _m.fit(
                    x_train,
                    y_train,
                    eval_set=[(x_valid, y_valid)],
                    verbose=self.config.verbose
                )
            
                ypred_temp = _m.predict_proba(x_valid)[:, 1]
                ypred.append(ypred_temp)
            ypred = np.column_stack(ypred)

            if self.config.use_predict_proba:
                ypred = self.model.predict_proba(x_valid)
            else:
                ypred = self.model.predict(x_valid)

            # calculate metric
            metric_dict = metrics.calculate(y_valid, ypred)
            scores.append(metric_dict)
            if self.config.only_one_train is True:
                break

        self.model_save(self.model)
        mean_metrics = self.dict_mean(scores)
        logger.info(f"Result metric score: {mean_metrics}")
        
        if test_df is None:
            return mean_metrics, None
        
        ypred = self.predict(test_df, targets)
        if self.config.use_predict_proba:
            ypred = self.model.predict_proba(test_df)
        else:
            ypred = self.model.predict(test_df)
            
        return {
            "training_metric": mean_metrics,
            "prediction": ypred,
        }
    
    def predict(
        self, 
        test_df: pd.DataFrame,
        targets: List[str],
        path: str = None,
    ) -> np.ndarray:
        model = self.model
        if path is not None:
            model = self.model_load()
        
        models = [model] * len(targets)
        ypred = []
        for idx, _m in enumerate(models):
            ypred_temp = _m.predict_proba(test_df)[:, 1]
            ypred.append(ypred_temp)
        ypred = np.column_stack(ypred)
        return ypred
    
    def model_save(self, model):
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{self.config.model_name}.pickle", "wb") as f:
            pickle.dump(model, f)
            
    def model_load(self) -> BaseModel:
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{self.config.model_name}.pickle", "rb") as f:
            model = pickle.load(f)
        return model
    
    def dict_mean(self, dict_list):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        return mean_dict
    
    def created_fold(
        self,
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
        if problem in ["binary_classification", "multi_class_classification"]:
            y = df[target_columns].values
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            for fold, (_, valid_indicies) in enumerate(skf.split(X=df, y=y)):
                df.loc[valid_indicies, "fold_id"] = fold
                
        elif problem in ["single_column_regression"]:
            y = df[target_columns].values
            num_bins = min(int(np.floor(1 + np.log2(len(df)))), 10)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            df["bins"] = pd.cut(df[target_columns].values.ravel(), bins=num_bins, labels=False)
            for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=df.bins.values)):
                df.loc[valid_indicies, "fold_id"] = fold
            df = df.drop("bins", axis=1)
            
        elif problem in ["multi_column_regression"]:
            # Todo: MultilabelStratifiedKFold
            y = df[target_columns].values
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=y)):
                df.loc[valid_indicies, "fold_id"] = fold
        
        elif problem in ["multi_label_classification"]:
            # Todo: MultilabelStratifiedKFold
            y = df[target_columns].values
            kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            for fold, (_, valid_indicies) in enumerate(kf.split(X=df, y=y)):
                df.loc[valid_indicies, "fold_id"] = fold

        else:
            raise Exception("Invalid problem type for created fold.")
            
        return df
    
    def check_problem_type(
        self,
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
    
    def feature_importacne(self):
        return self.model.feature_importances(columns=self.columns)
        