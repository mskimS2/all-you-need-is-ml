import os
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from typing import List, Dict, Any

from type import Problem
from const import Const
from metrics import Metric
from template.models import classifiers


@dataclass
class ClassifierTrainer:
    def set_up(
        self,
        models: List[Any] = None,
        optimize_hyperparams_tune: bool =  False,
    ):
        if models is None:
            self.models = self._get_models(optimize_hyperparams_tune)
        
        for _m in models:
            if not isinstance(_m, Pipeline):
                _m = Pipeline([_m.__repr__, _m])

        self.models = models

    def fit(
        self,         
        train_df: pd.DataFrame = None,
        features: List[str] = None,
        targets: List[str] = None,
        params: Dict[str, Any] = None,
        problem_type: Problem = None,
        num_classes: int = None,
    ):
        metrics = Metric(problem_type)
        train_df = self.created_fold(
            train_df, 
            self.config.problem_type, 
            self.config.num_folds, 
            self.config.shuffle, 
            self.config.random_seed, 
            targets,
        )
        
        for fold_id in range(self.config.num_folds):
            x_train = train_df[train_df[Const.FOLD_ID]!=fold_id][features]
            y_train = train_df[train_df[Const.FOLD_ID]!=fold_id][targets]
            x_valid = train_df[train_df[Const.FOLD_ID]!=fold_id][features]
            y_valid = train_df[train_df[Const.FOLD_ID]!=fold_id][targets]
            
            y_pred = {m.__repr__: []*len(targets) for m in self.models}
            scores = {m.__repr__: [] for m in self.models}
            for idx, _m in enumerate(self.models):
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
            
                if self.config.get("use_predict_proba") is not None:
                    pred = self.model.predict_proba(X=x_valid, **vars(self.config))
                else:
                    pred = self.model.predict(X=x_valid, **vars(self.config))
                
                # calculate metric
                metric_dict = metrics.calculate(y_valid, y_pred)
                scores[_m.__repr__].append(metric_dict)
                y_pred[_m.__repr__].append(pred)
                self.model_save(self.model, _m.__repr__+f"_{fold_id}")
                
                if self.config.only_one_train is True:
                    break
            
            y_pred = np.column_stack(y_pred)
        
        return pd.DataFrame.from_dict(
            {k:self.score_mean(s) for k, s in scores.items()},
            columns=["model"]+[f"{t}_pred" for t in targets]
        )
    
    def predict(
        self, 
        models: List[Any],
        test_df: pd.DataFrame = None,
        targets: List[str] = None,
    ) -> Dict[np.ndarray, Any]:
        if models is None:
            assert "models is None"
            
        y_pred = {m.__repr__: []* len(targets) for m in range(len(models))}
        for idx, _m in enumerate(models):
            if models.hasattr("predict_proba"):
                pred = _m.predict_proba(test_df)[:, 1]
            elif models.hasattr("predict"):
                pred = _m.predict(test_df)
            y_pred[_m.__repr__].append(pred)
        
        return {k: np.column_stack(v) for k, v in y_pred.items()}
    
    def save_model(self, model, name: str) -> None:
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{name}.pkl", "wb") as f:
            pickle.dump(model, f)
            
    def load_model(self) -> Any:
        os.makedirs(self.config.output_path, exist_ok=True)
        with open(f"{self.config.output_path}/{self.config.model_name}.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    
    def score_mean(self, dict_list):
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
        
    def _get_models(self, optimize_hyperparams_tune: bool =  False):
        return {
            "lr": classifiers.LogisticRegressionClassifierTemplate(),
            "knn": classifiers.KNeighborsClassifierTemplate(),
            "nb": classifiers.GaussianNBClassifierTemplate(),
            "dt": classifiers.DecisionTreeClassifierTemplate(),
            "svm": classifiers.SGDClassifierTemplate(),
            "rbfsvm": classifiers.SVCClassifierTemplate(),
            "gpc": classifiers.GaussianProcessClassifierTemplate(),
            "mlp": classifiers.MLPClassifierTemplate(),
            "ridge": classifiers.RidgeClassifierTemplate(),
            "rf": classifiers.RandomForestClassifierTemplate(),
            "qda": classifiers.QuadraticDiscriminantAnalysisTemplate(),
            "ada": classifiers.AdaBoostClassifierTemplate(),
            "gbc": classifiers.GradientBoostingClassifierTemplate(),
            "lda": classifiers.LinearDiscriminantAnalysisTemplate(),
            "et": classifiers.ExtraTreesClassifierTemplate(),
            "xgboost": classifiers.XGBClassifierTemplate(),
            "lightgbm": classifiers.LGBMClassifierTemplate(),
            "catboost": classifiers.CatBoostClassifierTemplate(),
            "dummy": classifiers.DummyClassifierTemplate(),
            "bagging": classifiers.BaggingClassifierTemplate(),
            "stacking": classifiers.StackingClassifierTemplate(),
            "voting": classifiers.VotingClassifierTemplate(),
            "calibratedCV": classifiers.CalibratedClassifierCVTemplate(),
        }