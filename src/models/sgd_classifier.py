import numpy as np
import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Dict, List
from const import Const
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder

@dataclass
class SgdClassifier(BaseModel):
    model: SGDClassifier
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self, *args, **kwargs):
        self.model = SGDClassifier(
            loss=kwargs.get("loss", self.config.loss),
            penalty=kwargs.get("penalty", self.config.penalty),
            alpha=kwargs.get("alpha", self.config.alpha),
            l1_ratio=kwargs.get("l1_ratio", self.config.l1_ratio),
            fit_intercept=kwargs.get("fit_intercept", self.config.fit_intercept),
            max_iter=kwargs.get("max_iter", self.config.max_iter),
            tol=kwargs.get("tol", self.config.tol),
            shuffle=kwargs.get("shuffle", self.config.shuffle),
            verbose=kwargs.get("verbose", self.config.verbose),
            epsilon=kwargs.get("epsilon", self.config.epsilon),
            n_jobs=kwargs.get("n_jobs", self.config.n_jobs),
            random_state=kwargs.get("random_state", self.config.random_state),
            learning_rate=kwargs.get("learning_rate", self.config.learning_rate),
            eta0=kwargs.get("eta0", self.config.eta0),
            power_t=kwargs.get("power_t", self.config.power_t),
            early_stopping=kwargs.get("early_stopping", self.config.early_stopping),
            validation_fraction=kwargs.get("validation_fraction", self.config.validation_fraction),
            n_iter_no_change=kwargs.get("n_iter_no_change", self.config.n_iter_no_change),
            class_weight=kwargs.get("class_weight", self.config.class_weight),
            warm_start=kwargs.get("warm_start", self.config.warm_start),
            average=kwargs.get("average", self.config.average),
        )
    
    def fit(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        y=kwargs.get("y")
        if y is None:
            raise ValueError("y is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype("float32")
                    
        return self.model.fit(
            X=x.values,
            y=y,
            sample_weight=kwargs.get("sample_weight"),
            coef_init=kwargs.get("coef_init"), 
            intercept_init=kwargs.get("intercept_init"),
        )
    
    def predict(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype("float32")
        
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes is None")
        
        oh = OneHotEncoder(sparse=False).fit([[i] for i in range(num_classes)])
        pred = self.model.predict(X=x.values)
        return oh.transform(pred.reshape(-1, 1))
    
    def predict_proba(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        if isinstance(x, pd.DataFrame):
            for col in x.columns:
                if x[col].dtypes != np.float32:
                    x[col] = x[col].astype(np.float32)
        
        return self.model.predict_proba(X=x.values)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("lime") is not None:
            # TODO: lime value
            pass
        
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")
        
        return None
    
    def optimize_hyper_params(
        self, 
        df: pd.DataFrame,
        features: List[str],
        targets: List[str], 
        **hparams: Dict,
    ):
        config = {
           "loss": hparams.get("loss", self.config.loss),
           "penalty": hparams.get("penalty", self.config.penalty),
           "alpha": hparams.get("alpha", self.config.alpha),
           "l1_ratio": hparams.get("l1_ratio", self.config.l1_ratio),
           "fit_intercept": hparams.get("fit_intercept", self.config.fit_intercept),
           "max_iter": hparams.get("max_iter", self.config.max_iter),
           "tol": hparams.get("tol", self.config.tol),
           "shuffle": hparams.get("shuffle", self.config.shuffle),
           "verbose": hparams.get("verbose", self.config.verbose),
           "epsilon": hparams.get("epsilon", self.config.epsilon),
           "n_jobs": hparams.get("n_jobs", self.config.n_jobs),
           "random_state": hparams.get("random_state", self.config.random_state),
           "learning_rate": hparams.get("learning_rate", self.config.learning_rate),
           "eta0": hparams.get("eta0", self.config.eta0),
           "power_t": hparams.get("power_t", self.config.power_t),
           "early_stopping": hparams.get("early_stopping", self.config.early_stopping),
           "validation_fraction": hparams.get("validation_fraction", self.config.validation_fraction),
           "n_iter_no_change": hparams.get("n_iter_no_change", self.config.n_iter_no_change),
           "class_weight": hparams.get("class_weight", self.config.class_weight),
           "warm_start": hparams.get("warm_start", self.config.warm_start),
           "average": hparams.get("average", self.config.average),
        }
        
        model = SGDClassifier(**config)
        
        accuaraies = []
        for fold in range(self.config.num_folds):
            x_train, y_train = df[df[Const.FOLD_ID]!=fold][features], df[df[Const.FOLD_ID]!=fold][targets]
            x_valid, y_valid = df[df[Const.FOLD_ID]!=fold][features], df[df[Const.FOLD_ID]!=fold][targets]

            model.fit(
                X=x_train,
                y=y_train,
                eval_set=[(x_valid, y_valid)],
                **config,
            )
            
            if self.config.use_predict_proba:
                y_pred = self.model.predict_proba(X=x_valid)
            else:
                y_pred = self.model.predict(X=x_valid)
            accuaraies.append(metrics.accuracy_score(y_valid, y_pred))

        return -1.0 * np.mean(accuaraies)