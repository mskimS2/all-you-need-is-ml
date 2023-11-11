import pandas as pd
import numpy as np
from models.base import BaseModel
from dataclasses import dataclass
from typing import Union, Dict
from lightgbm import LGBMClassifier, LGBMRegressor


@dataclass
class LGBM(BaseModel):
    model: Union[LGBMClassifier, LGBMRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self):
        self.model.num_iterations = self.config.num_iterations
        self.model.num_leaves = self.config.num_leaves
        self.model.learning_rate = self.config.learning_rate
        self.model.tree_learner = self.config.tree_learner
        self.model.num_threads = self.config.num_threads
        self.model.device_type = self.config.device
        self.model.seed = self.config.random_seed
        self.model.deterministic = self.config.deterministic
        self.model.early_stopping_round = self.config.early_stopping_round
        self.model.force_col_wise = self.config.force_col_wise
        self.model.force_row_wise = self.config.force_row_wise
        self.model.histogram_pool_size = self.config.histogram_pool_size
        self.model.max_depth = self.config.max_depth
        self.model.min_data_in_leaf = self.config.min_data_in_leaf
        self.model.min_sum_hessian_in_leaf = self.config.min_sum_hessian_in_leaf
        self.model.bagging_fraction = self.config.bagging_fraction
        self.model.pos_bagging_fraction = self.config.pos_bagging_fraction
        self.model.neg_bagging_fraction = self.config.neg_bagging_fraction
        self.model.bagging_freq = self.config.bagging_freq
        self.model.bagging_seed = self.config.bagging_seed
        self.model.feature_fraction = self.config.feature_fraction
        self.model.feature_fraction_bynode = self.config.feature_fraction_bynode
        self.model.feature_fraction_seed = self.config.feature_fraction_seed
        self.model.extra_trees = self.config.extra_trees
        self.model.extra_trees = self.config.extra_trees
        self.model.first_metric_only = self.config.first_metric_only
        self.model.max_delta_step = self.config.max_delta_step
        self.model.lambda_l1 = self.config.lambda_l1
        self.model.lambda_l2 = self.config.lambda_l2
        self.model.linear_lambda = self.config.linear_lambda
        self.model.min_gain_to_split = self.config.min_gain_to_split
        self.model.drop_rate = self.config.drop_rate
        self.model.max_drop = self.config.max_drop
        self.model.skip_drop = self.config.skip_drop
        self.model.xgboost_dart_mode = self.config.xgboost_dart_mode
        self.model.uniform_drop = self.config.uniform_drop
        self.model.drop_seed = self.config.drop_seed
        self.model.top_rate = self.config.top_rate
        self.model.other_rate = self.config.other_rate
        self.model.min_data_per_group = self.config.min_data_per_group
        self.model.max_cat_threshold = self.config.max_cat_threshold
        self.model.cat_l2 = self.config.cat_l2
        self.model.cat_smooth = self.config.cat_smooth
        self.model.max_cat_to_onehot = self.config.max_cat_to_onehot
        self.model.top_k = self.config.top_k
        self.model.monotone_constraints = self.config.monotone_constraints
        self.model.monotone_constraints_method = self.config.monotone_constraints_method
        self.model.monotone_penalty = self.config.monotone_penalty
        self.model.feature_contri = self.config.feature_contri
        self.model.forcedsplits_filename = self.config.forcedsplits_filename
        self.model.interaction_constraints = self.config.interaction_constraints
        self.model.refit_decay_rate = self.config.refit_decay_rate
        self.model.cegb_tradeoff = self.config.cegb_tradeoff
        self.model.cegb_penalty_split = self.config.cegb_penalty_split
        self.model.output_model = self.config.output_model
        self.model.saved_feature_importance_type = self.config.saved_feature_importance_type
        self.model.path_smooth = self.config.path_smooth
        self.model.verbosity = self.config.verbosity
        self.model.snapshot_freq = self.config.snapshot_freq
        self.model.use_quantized_grad = self.config.use_quantized_grad
        self.model.num_grad_quant_bins = self.config.num_grad_quant_bins
        self.model.quant_train_renew_leaf = self.config.quant_train_renew_leaf
        self.model.stochastic_rounding = self.config.stochastic_rounding
    
    def fit(self, *args, **kwargs):
        return self.model.fit(
            X=kwargs["X"],
            y=kwargs["y"],
            eval_set=kwargs["eval_set"],
            # *args, **kwargs,
        )
    
    def predict(self, *args, **kwargs):
        return self.model.predict(
            X=kwargs["X"],
            # *args, **kwargs,
        )
    
    def predict_proba(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)
    
    def feature_importances(self, *args, **kwargs) -> pd.DataFrame:
        if kwargs.get("shap") is not None:
            # TODO: shap value
            pass
        
        if kwargs.get("columns") is None:
            raise ValueError("Train_df columns is None")
        
        assert len(kwargs["columns"]) == len(self.model.feature_importances_)
        
        return pd.DataFrame.from_dict(
            {c: [v] for c, v in zip(kwargs["columns"], self.model.feature_importances_)}, 
            columns=["feature_importance"],
            orient="index",
        )