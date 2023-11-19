import os
import pickle
import numpy as np
import logging
import pandas as pd
import optuna
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.model_selection import KFold, StratifiedKFold

import utils
from models.base import BaseModel
from preprocessor import Encoder, Scaler, Preprocessor
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
        reduce_memory_usage=True,
        optimize_hyperparams=None,
    ):
        logger.info(f"configuration: {self.model.config}")
        self.columns = features
        metrics = Metric(self.config.problem_type)
        scores = []
        
        if reduce_memory_usage:
            if train_df is not None:
                train_df = utils.reduce_memory_usage(train_df)
            if test_df is not None:
                test_df = utils.reduce_memory_usage(test_df)
                
        problem_type, num_classes = self.check_problem_type(train_df, self.config.task, targets)
        self.config.problem_type = problem_type
        self.config.num_classes = num_classes
        logger.info(f"problem type: {self.config.problem_type}, detected labels: {self.config.num_classes}")
        
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
        
        if optimize_hyperparams is not None:
            self.optimize_hyper_parameters(
                df=train_df,
                features=features,
                targets=targets,
                hparams=optimize_hyperparams,
            )
        
        for fold in range(self.config.num_folds):
            x_train = train_df[train_df[Const.FOLD_ID]!=fold][features]
            y_train = train_df[train_df[Const.FOLD_ID]!=fold][targets]
            x_valid = train_df[train_df[Const.FOLD_ID]!=fold][features]
            y_valid = train_df[train_df[Const.FOLD_ID]!=fold][targets]

            y_pred = []
            models = [self.model] * len(targets)
            for idx, _m in enumerate(models):
                _m.fit(
                    X=x_train,
                    y=y_train,
                    eval_set=[(x_valid, y_valid)],
                    verbose=self.config.verbose,
                    check_input=True,
                    validate_features=True,
                    raw_score=False,
                    output_margin=False,
                    kwargs=self.config,
                )
            
                if self.config.use_predict_proba:
                    y_pred_temp = self.model.predict_proba(X=x_valid, **vars(self.config))
                else:
                    y_pred_temp = self.model.predict(X=x_valid, **vars(self.config))
                    
                y_pred.append(y_pred_temp)
            y_pred = np.column_stack(y_pred)
            
            # calculate metric
            metric_dict = metrics.calculate(y_valid, y_pred)
            scores.append(metric_dict)
            if self.config.only_one_train is True:
                break

        self.model_save(self.model)
        mean_metrics = self.dict_mean(scores)
        logger.info(f"`{self.model.config.model_name}` model metric score: {mean_metrics}")
        
        res = {
            "training_metric": mean_metrics,
        }
        
        if test_df is not None:
            if self.config.use_predict_proba:
                y_pred = self.model.predict_proba(X=test_df[features], **vars(self.config))
            else:
                y_pred = self.model.predict(X=test_df[features], **vars(self.config))
            res["prediction"] = y_pred
        
        return res
    
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
        y_pred = []
        for idx, _m in enumerate(models):
            y_pred_temp = _m.predict_proba(test_df)[:, 1]
            y_pred.append(y_pred_temp)
        y_pred = np.column_stack(y_pred)
        return y_pred
    
    def model_save(self, model) -> None:
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{self.config.model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
            
    def model_load(self) -> BaseModel:
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{self.config.model_name}.pkl", "rb") as f:
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
        if Const.FOLD_ID in df.columns:
            logger.info("`fold_id` column already exists from input dataframe.")
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
            skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
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
    
    def feature_importacne(self):
        return self.model.feature_importances(columns=self.columns)
        
    def optimize_hyper_parameters(
        self, 
        df: pd.DataFrame, 
        features: List[str] = None,
        targets: List[str] = None,
        n_trials: int = 15,
        direction: str = "minimize",
    ):
        logger.info(f"optimize hyperparameters")
        
        dir = -1 if direction == "minimize" else 1
             
        study = optuna.create_study(direction=direction)
        study.optimize(
            partial(
                self.model.optimize_hyper_params*dir, 
                df=df, 
                features=features, 
                targets=targets,
            ), 
            n_trials=n_trials,
        )
        self.model.set_up(**study.best_params)
        logger.info(f"best parameters: ", **study.best_params)