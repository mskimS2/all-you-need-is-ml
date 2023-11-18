import pandas as pd
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict
from catboost import CatBoostClassifier, CatBoostRegressor


@dataclass
class CatBoost(BaseModel):
    model: Union[CatBoostClassifier, CatBoostRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.iterations=self.config.iterations
        self.model.learning_rate=self.config.learning_rate
        self.model.early_stopping_rounds=self.config.early_stopping_rounds
        self.model.random_seed=self.config.random_seed
        self.model.l2_leaf_reg=self.config.l2_leaf_reg
        self.model.bootstrap_type=self.config.bootstrap_type
        self.model.bagging_temperature=self.config.bagging_temperature
        self.model.subsample=self.config.subsample
        self.model.sampling_frequency=self.config.sampling_frequency
        self.model.sampling_unit=self.config.sampling_unit
        self.model.mvs_reg=self.config.mvs_reg
        self.model.random_strength=self.config.random_strength
        self.model.use_best_model=self.config.use_best_model
        self.model.best_model_min_trees=self.config.best_model_min_trees
        self.model.depth=self.config.depth
        self.model.grow_policy=self.config.grow_policy
        self.model.min_data_in_leaf=self.config.min_data_in_leaf
        self.model.max_leaves=self.config.max_leaves
        self.model.ignore_features=self.config.ignore_features
        self.model.scale_pos_weight=self.config.scale_pos_weight
        self.model.one_hot_max_size=self.config.one_hot_max_size
        self.model.has_time=self.config.has_time
        self.model.rsm=self.config.rsm
        self.model.nan_mode=self.config.nan_mode
        self.model.input_borders=self.config.input_borders
        self.model.output_borders=self.config.output_borders
        self.model.fold_permutation_block=self.config.fold_permutation_block
        self.model.leaf_estimation_method=self.config.leaf_estimation_method
        self.model.leaf_estimation_iterations=self.config.leaf_estimation_iterations
        self.model.leaf_estimation_backtracking=self.config.leaf_estimation_backtracking
        self.model.fold_len_multiplier=self.config.fold_len_multiplier
        self.model.approx_on_full_history=self.config.approx_on_full_history
        self.model.boosting_type=self.config.boosting_type
        self.model.boost_from_average=self.config.boost_from_average
        self.model.langevin=self.config.langevin
        self.model.diffusion_temperature=self.config.diffusion_temperature
        self.model.posterior_sampling=self.config.posterior_sampling
        self.model.allow_const_label=self.config.allow_const_label
        self.model.class_weights=self.config.class_weights
        self.model.class_name=self.config.class_name
        self.model.auto_class_weights=self.config.auto_class_weights
        self.model.score_function=self.config.score_function
        self.model.task_type=self.config.device
    
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
        
        return self.model.predict(
            data=x, 
            prediction_type=kwargs.get("prediction_type", "Class"),
            ntree_start=kwargs.get("ntree_start", 0),
            ntree_end=kwargs.get("ntree_end", 0), 
            thread_count=kwargs.get("thread_count", -1), 
            verbose=kwargs.get("verbose"), 
            task_type=kwargs.get("task_type", "CPU"), 
        )
    
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