import pandas as pd
import numpy as np
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict, List
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder

from const import Const


@dataclass
class CatBoostClassifier(BaseModel):
    model: CatBoostClassifier
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self, *args, **kwargs):
        for k, v in vars(self.config).items():
            setattr(self.model, k, v)
            
        if kwargs is not None:
            for k, v in kwargs.items():
                setattr(self.model, k, v)
    
    def fit(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        y=kwargs.get("y")
        if y is None:
            raise ValueError("y is None")
        
        return self.model.fit(
            X=x,
            y=y, 
            cat_features=kwargs.get("cat_features"), 
            text_features=kwargs.get("text_features"), 
            embedding_features=kwargs.get("embedding_features"), 
            sample_weight=kwargs.get("sample_weight"), 
            baseline=kwargs.get("baseline"), 
            use_best_model=kwargs.get("use_best_model"),
            eval_set=kwargs.get("eval_set"), 
            verbose=kwargs.get("verbose"), 
            logging_level=kwargs.get("logging_level"), 
            plot=kwargs.get("plot"), 
            plot_file=kwargs.get("plot_file"), 
            column_description=kwargs.get("column_description"),
            verbose_eval=kwargs.get("verbose_eval"), 
            metric_period=kwargs.get("metric_period"), 
            silent=kwargs.get("silent"), 
            early_stopping_rounds=kwargs.get("early_stopping_rounds"),
            save_snapshot=kwargs.get("save_snapshot"), 
            snapshot_file=kwargs.get("snapshot_file"),
            snapshot_interval=kwargs.get("snapshot_interval"), 
            init_model=kwargs.get("init_model"), 
            callbacks=kwargs.get("callbacks"),
        )
    
    def predict(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        num_classes = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError("num_classes is None")
        
        oh = OneHotEncoder(sparse=False).fit([[i] for i in range(num_classes)])
        pred = self.model.predict(
            data=x, 
            prediction_type=kwargs.get("prediction_type", "Class"),
            ntree_start=kwargs.get("ntree_start", 0),
            ntree_end=kwargs.get("ntree_end", 0), 
            thread_count=kwargs.get("thread_count", -1), 
            verbose=kwargs.get("verbose"), 
            task_type=kwargs.get("task_type", "CPU"), 
        )
        return oh.transform(pred.reshape(-1, 1))
    
    def predict_proba(self, *args, **kwargs):
        x=kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict_proba(
            X=x, 
            ntree_start=kwargs.get("ntree_start", 0),
            ntree_end=kwargs.get("ntree_end", 0), 
            thread_count=kwargs.get("thread_count", -1), 
            verbose=kwargs.get("verbose"), 
            task_type=kwargs.get("task_type", "CPU"), 
        )
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")
        
        assert len(kwargs["columns"]) == len(self.model.feature_importances_)
        
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("lime") is not None:
            # TODO: lime value
            pass
        
        return pd.DataFrame.from_dict(
            {c: [v] for c, v in zip(kwargs["columns"], self.model.feature_importances_)}, 
            columns=["feature_importance"],
            orient="index",
        )
        
    def optimize_hyper_params(
        self, 
        df: pd.DataFrame,
        features: List[str],
        targets: List[str], 
        **hparams: Dict,
    ):
        config = {
            "iterations": hparams.get("iterations", self.config.iterations),
            "learning_rate": hparams.get("learning_rate", self.config.learning_rate),
            "early_stopping_rounds": hparams.get("early_stopping_rounds", self.config.early_stopping_rounds),
            "random_seed": hparams.get("random_seed", self.config.random_seed),
            "l2_leaf_reg": hparams.get("l2_leaf_reg", self.config.l2_leaf_reg),
            "bootstrap_type": hparams.get("bootstrap_type", self.config.bootstrap_type),
            "bagging_temperature": hparams.get("bagging_temperature", self.config.bagging_temperature),
            "bagging_temperature": hparams.get("bagging_temperature", self.config.subsample),
            "sampling_frequency": hparams.get("sampling_frequency", self.config.sampling_frequency),
            "sampling_unit": hparams.get("sampling_unit", self.config.sampling_unit),
            "mvs_reg": hparams.get("mvs_reg", self.config.mvs_reg),
            "random_strength": hparams.get("random_strength", self.config.random_strength),
            "use_best_model": hparams.get("use_best_model", self.config.use_best_model),
            "best_model_min_trees": hparams.get("best_model_min_trees", self.config.best_model_min_trees),
            "depth": hparams.get("depth", self.config.depth),
            "grow_policy": hparams.get("grow_policy", self.config.grow_policy),
            "min_data_in_leaf": hparams.get("min_data_in_leaf", self.config.min_data_in_leaf),
            "max_leaves": hparams.get("max_leaves", self.config.max_leaves),
            "scale_pos_weight": hparams.get("scale_pos_weight", self.config.scale_pos_weight),
            "one_hot_max_size": hparams.get("one_hot_max_size", self.config.one_hot_max_size),
            "has_time": hparams.get("has_time", self.config.has_time),
            "rsm": hparams.get("rsm", self.config.rsm),
            "nan_mode": hparams.get("nan_mode", self.config.nan_mode),
            "input_borders": hparams.get("input_borders", self.config.input_borders),
            "output_borders": hparams.get("output_borders", self.config.output_borders),
            "fold_permutation_block": hparams.get("fold_permutation_block", self.config.fold_permutation_block),
            "leaf_estimation_method": hparams.get("leaf_estimation_method", self.config.leaf_estimation_method),
            "leaf_estimation_iterations": hparams.get("leaf_estimation_iterations", self.config.leaf_estimation_iterations),
            "leaf_estimation_backtracking": hparams.get("leaf_estimation_backtracking", self.config.leaf_estimation_backtracking),
            "fold_len_multiplier": hparams.get("fold_len_multiplier", self.config.fold_len_multiplier),
            "approx_on_full_history": hparams.get("approx_on_full_history", self.config.approx_on_full_history),
            "boosting_type": hparams.get("boosting_type", self.config.boosting_type),
            "boost_from_average": hparams.get("boost_from_average", self.config.boost_from_average),
            "langevin": hparams.get("langevin", self.config.langevin),
            "diffusion_temperature": hparams.get("diffusion_temperature", self.config.diffusion_temperature),
            "posterior_sampling": hparams.get("posterior_sampling", self.config.posterior_sampling),
            "allow_const_label": hparams.get("allow_const_label", self.config.allow_const_label),
            "class_weights": hparams.get("class_weights", self.config.class_weights),
            "auto_class_weights": hparams.get("auto_class_weights", self.config.auto_class_weights),
            "score_function": hparams.get("score_function", self.config.score_function),
            "device": hparams.get("device", self.config.device),
        }
        
        if isinstance(self.model, CatBoostClassifier):
            model = CatBoostClassifier(**config)
        
        accuaraies = []
        for fold in range(self.config.num_folds):
            x_train = df[df[Const.FOLD_ID]!=fold][features]
            y_train = df[df[Const.FOLD_ID]!=fold][targets]
            x_valid = df[df[Const.FOLD_ID]!=fold][features]
            y_valid = df[df[Const.FOLD_ID]!=fold][targets]

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