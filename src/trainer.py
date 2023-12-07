import os
import pickle
import numpy as np
import pandas as pd
import optuna
from omegaconf import DictConfig
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple

from type import Problem, Task
from const import Const
from preprocessor import Encoder, Scaler, Preprocessor
from metrics import Metric


@dataclass
class Trainer:
    
    def fit(
        self, 
        train_df: pd.DataFrame = None,
        test_df: pd.DataFrame = None,
        features: List[str] = None,
        targets: List[str] = None,
        optimize_hyperparams = None,
        problem_type: Problem = None,
        num_classes: int = None,
    ):
        print(f"configuration: {self.model.config}")
        self.columns = features
        scores = []
        
        metrics = Metric(problem_type)
        
        train_df = self.created_fold(
            train_df, 
            self.config.problem_type, 
            self.config.num_folds, 
            self.config.shuffle, 
            self.config.random_seed, 
            targets,
        )
        
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
            
                if self.config.use_predict_proba is not None:
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
        print(f"`{self.model.config.model_name}` model metric score: {mean_metrics}")
        
        res = {
            "training_metric": mean_metrics,
        }
        
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
    
    def save_model(self, model) -> None:
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{self.config.model_name}.pkl", "wb") as f:
            pickle.dump(model, f)
            
    def load_model(self):
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{self.config.model_name}.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    
    def dict_mean(self, dict_list):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        return mean_dict
    
    def feature_importance(self):
        return self.model.feature_importances(
            columns=self.columns, 
            problem_type=self.problem_type, 
            num_classes=self.num_classes, 
        )
        
    def optimize_hyper_parameters(
        self, 
        df: pd.DataFrame, 
        features: List[str] = None,
        targets: List[str] = None,
        n_trials: int = 15,
        direction: str = "minimize",
    ):             
        study = optuna.create_study(direction=direction)
        study.optimize(
            partial(
                self.model.optimize_hyper_params, 
                df=df, 
                features=features, 
                targets=targets,
            ), 
            n_trials=n_trials,
        )
        
        for k, v in study.best_params.items():
            setattr(self.model.config, k, v)