import pandas as pd
import numpy as np
from models.base import BaseModel
from dataclasses import dataclass
from typing import Dict, List, Union
from sklearn import metrics
from lightgbm import LGBMClassifier, LGBMRegressor

from const import Const

@dataclass
class LightGBM(BaseModel):
    model: Union[LGBMClassifier, LGBMRegressor]
    config: Dict
    
    def __post_init__(self):
        self.set_up()
    
    def set_up(self, *args, **kwargs):
        if isinstance(self.model, LGBMClassifier):
            self.model = LGBMClassifier(
                num_iterations=kwargs.get("num_iterations", self.config.num_iterations),
                num_leaves=kwargs.get("num_leaves", self.config.num_leaves),
                learning_rate=kwargs.get("learning_rate", self.config.learning_rate),
                tree_learner=kwargs.get("tree_learner", self.config.tree_learner),
                num_threads=kwargs.get("num_threads", self.config.num_threads),
                device_type=kwargs.get("device_type", self.config.device),
                seed=kwargs.get("seed", self.config.random_seed),
                deterministic=kwargs.get("deterministic", self.config.deterministic),
                early_stopping_round=kwargs.get("early_stopping_round", self.config.early_stopping_round),
                force_col_wise=kwargs.get("force_col_wise", self.config.force_col_wise),
                force_row_wise=kwargs.get("force_row_wise", self.config.force_row_wise),
                histogram_pool_size=kwargs.get("histogram_pool_size", self.config.histogram_pool_size),
                max_depth=kwargs.get("max_depth", self.config.max_depth),
                min_data_in_leaf=kwargs.get("min_data_in_leaf", self.config.min_data_in_leaf),
                min_sum_hessian_in_leaf=kwargs.get("min_sum_hessian_in_leaf", self.config.min_sum_hessian_in_leaf),
                bagging_fraction=kwargs.get("bagging_fraction", self.config.bagging_fraction),
                pos_bagging_fraction=kwargs.get("pos_bagging_fraction", self.config.pos_bagging_fraction),
                neg_bagging_fraction=kwargs.get("neg_bagging_fraction", self.config.neg_bagging_fraction),
                bagging_freq=kwargs.get("bagging_freq", self.config.bagging_freq),
                bagging_seed=kwargs.get("bagging_seed", self.config.bagging_seed),
                feature_fraction=kwargs.get("feature_fraction", self.config.feature_fraction),
                feature_fraction_bynode=kwargs.get("feature_fraction_bynode", self.config.feature_fraction_bynode),
                feature_fraction_seed=kwargs.get("feature_fraction_seed", self.config.feature_fraction_seed),
                extra_trees=kwargs.get("extra_trees", self.config.extra_trees),
                first_metric_only=kwargs.get("first_metric_only", self.config.first_metric_only),
                max_delta_step=kwargs.get("max_delta_step", self.config.max_delta_step),
                lambda_l1=kwargs.get("lambda_l1", self.config.lambda_l1),
                lambda_l2=kwargs.get("lambda_l2", self.config.lambda_l2),
                linear_lambda=kwargs.get("linear_lambda", self.config.linear_lambda),
                min_gain_to_split=kwargs.get("min_gain_to_split", self.config.min_gain_to_split),
                drop_rate=kwargs.get("drop_rate", self.config.drop_rate),
                max_drop=kwargs.get("max_drop", self.config.max_drop),
                skip_drop=kwargs.get("skip_drop", self.config.skip_drop),
                xgboost_dart_mode=kwargs.get("xgboost_dart_mode", self.config.xgboost_dart_mode),
                uniform_drop=kwargs.get("uniform_drop", self.config.uniform_drop),
                drop_seed=kwargs.get("drop_seed", self.config.drop_seed),
                top_rate=kwargs.get("top_rate", self.config.top_rate),
                other_rate=kwargs.get("other_rate", self.config.other_rate),
                min_data_per_group=kwargs.get("min_data_per_group", self.config.min_data_per_group),
                max_cat_threshold=kwargs.get("max_cat_threshold", self.config.max_cat_threshold),
                cat_l2=kwargs.get("cat_l2", self.config.cat_l2),
                cat_smooth=kwargs.get("cat_smooth", self.config.cat_smooth),
                max_cat_to_onehot=kwargs.get("max_cat_to_onehot", self.config.max_cat_to_onehot),
                top_k=kwargs.get("top_k", self.config.top_k),
                monotone_constraints=kwargs.get("monotone_constraints", self.config.monotone_constraints),
                monotone_constraints_method=kwargs.get("monotone_constraints_method", self.config.monotone_constraints_method),
                monotone_penalty=kwargs.get("monotone_penalty", self.config.monotone_penalty),
                feature_contri=kwargs.get("feature_contri", self.config.feature_contri),
                forcedsplits_filename=kwargs.get("forcedsplits_filename", self.config.forcedsplits_filename),
                interaction_constraints=kwargs.get("interaction_constraints", self.config.interaction_constraints),
                refit_decay_rate=kwargs.get("refit_decay_rate", self.config.refit_decay_rate),
                cegb_tradeoff=kwargs.get("cegb_tradeoff", self.config.cegb_tradeoff),
                cegb_penalty_split=kwargs.get("cegb_penalty_split", self.config.cegb_penalty_split),
                output_model=kwargs.get("output_model", self.config.output_model),
                saved_feature_importance_type=kwargs.get("saved_feature_importance_type", self.config.saved_feature_importance_type),
                path_smooth=kwargs.get("path_smooth", self.config.path_smooth),
                verbosity=kwargs.get("verbosity", self.config.verbosity),
                snapshot_freq=kwargs.get("snapshot_freq", self.config.snapshot_freq),
                use_quantized_grad=kwargs.get("use_quantized_grad", self.config.use_quantized_grad),
                num_grad_quant_bins=kwargs.get("num_grad_quant_bins", self.config.num_grad_quant_bins),
                quant_train_renew_leaf=kwargs.get("quant_train_renew_leaf", self.config.quant_train_renew_leaf),
                stochastic_rounding=kwargs.get("stochastic_rounding", self.config.stochastic_rounding),
            )
        elif isinstance(self.model, LGBMRegressor):
            self.model = LGBMRegressor(
                num_iterations=kwargs.get("num_iterations", self.config.num_iterations),
                num_leaves=kwargs.get("num_leaves", self.config.num_leaves),
                learning_rate=kwargs.get("learning_rate", self.config.learning_rate),
                tree_learner=kwargs.get("tree_learner", self.config.tree_learner),
                num_threads=kwargs.get("num_threads", self.config.num_threads),
                device_type=kwargs.get("device_type", self.config.device),
                seed=kwargs.get("seed", self.config.random_seed),
                deterministic=kwargs.get("deterministic", self.config.deterministic),
                early_stopping_round=kwargs.get("early_stopping_round", self.config.early_stopping_round),
                force_col_wise=kwargs.get("force_col_wise", self.config.force_col_wise),
                force_row_wise=kwargs.get("force_row_wise", self.config.force_row_wise),
                histogram_pool_size=kwargs.get("histogram_pool_size", self.config.histogram_pool_size),
                max_depth=kwargs.get("max_depth", self.config.max_depth),
                min_data_in_leaf=kwargs.get("min_data_in_leaf", self.config.min_data_in_leaf),
                min_sum_hessian_in_leaf=kwargs.get("min_sum_hessian_in_leaf", self.config.min_sum_hessian_in_leaf),
                bagging_fraction=kwargs.get("bagging_fraction", self.config.bagging_fraction),
                pos_bagging_fraction=kwargs.get("pos_bagging_fraction", self.config.pos_bagging_fraction),
                neg_bagging_fraction=kwargs.get("neg_bagging_fraction", self.config.neg_bagging_fraction),
                bagging_freq=kwargs.get("bagging_freq", self.config.bagging_freq),
                bagging_seed=kwargs.get("bagging_seed", self.config.bagging_seed),
                feature_fraction=kwargs.get("feature_fraction", self.config.feature_fraction),
                feature_fraction_bynode=kwargs.get("feature_fraction_bynode", self.config.feature_fraction_bynode),
                feature_fraction_seed=kwargs.get("feature_fraction_seed", self.config.feature_fraction_seed),
                extra_trees=kwargs.get("extra_trees", self.config.extra_trees),
                first_metric_only=kwargs.get("first_metric_only", self.config.first_metric_only),
                max_delta_step=kwargs.get("max_delta_step", self.config.max_delta_step),
                lambda_l1=kwargs.get("lambda_l1", self.config.lambda_l1),
                lambda_l2=kwargs.get("lambda_l2", self.config.lambda_l2),
                linear_lambda=kwargs.get("linear_lambda", self.config.linear_lambda),
                min_gain_to_split=kwargs.get("min_gain_to_split", self.config.min_gain_to_split),
                drop_rate=kwargs.get("drop_rate", self.config.drop_rate),
                max_drop=kwargs.get("max_drop", self.config.max_drop),
                skip_drop=kwargs.get("skip_drop", self.config.skip_drop),
                xgboost_dart_mode=kwargs.get("xgboost_dart_mode", self.config.xgboost_dart_mode),
                uniform_drop=kwargs.get("uniform_drop", self.config.uniform_drop),
                drop_seed=kwargs.get("drop_seed", self.config.drop_seed),
                top_rate=kwargs.get("top_rate", self.config.top_rate),
                other_rate=kwargs.get("other_rate", self.config.other_rate),
                min_data_per_group=kwargs.get("min_data_per_group", self.config.min_data_per_group),
                max_cat_threshold=kwargs.get("max_cat_threshold", self.config.max_cat_threshold),
                cat_l2=kwargs.get("cat_l2", self.config.cat_l2),
                cat_smooth=kwargs.get("cat_smooth", self.config.cat_smooth),
                max_cat_to_onehot=kwargs.get("max_cat_to_onehot", self.config.max_cat_to_onehot),
                top_k=kwargs.get("top_k", self.config.top_k),
                monotone_constraints=kwargs.get("monotone_constraints", self.config.monotone_constraints),
                monotone_constraints_method=kwargs.get("monotone_constraints_method", self.config.monotone_constraints_method),
                monotone_penalty=kwargs.get("monotone_penalty", self.config.monotone_penalty),
                feature_contri=kwargs.get("feature_contri", self.config.feature_contri),
                forcedsplits_filename=kwargs.get("forcedsplits_filename", self.config.forcedsplits_filename),
                interaction_constraints=kwargs.get("interaction_constraints", self.config.interaction_constraints),
                refit_decay_rate=kwargs.get("refit_decay_rate", self.config.refit_decay_rate),
                cegb_tradeoff=kwargs.get("cegb_tradeoff", self.config.cegb_tradeoff),
                cegb_penalty_split=kwargs.get("cegb_penalty_split", self.config.cegb_penalty_split),
                output_model=kwargs.get("output_model", self.config.output_model),
                saved_feature_importance_type=kwargs.get("saved_feature_importance_type", self.config.saved_feature_importance_type),
                path_smooth=kwargs.get("path_smooth", self.config.path_smooth),
                verbosity=kwargs.get("verbosity", self.config.verbosity),
                snapshot_freq=kwargs.get("snapshot_freq", self.config.snapshot_freq),
                use_quantized_grad=kwargs.get("use_quantized_grad", self.config.use_quantized_grad),
                num_grad_quant_bins=kwargs.get("num_grad_quant_bins", self.config.num_grad_quant_bins),
                quant_train_renew_leaf=kwargs.get("quant_train_renew_leaf", self.config.quant_train_renew_leaf),
                stochastic_rounding=kwargs.get("stochastic_rounding", self.config.stochastic_rounding),
            )
    
    def fit(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        y = kwargs.get("y")        
        if y is None:
            raise ValueError("y is None")
        
        return self.model.fit(
            X=x,
            y=y,
            eval_set=kwargs.get("eval_set"),
            sample_weight=kwargs.get("sample_weight"),
            init_score=kwargs.get("init_score"),
            eval_names=kwargs.get("eval_names"),
            eval_sample_weight=kwargs.get("eval_sample_weight"),
            eval_init_score=kwargs.get("eval_init_score"),
            eval_metric=kwargs.get("eval_metric"),
            feature_name=kwargs.get("feature_name"),
            categorical_feature=kwargs.get("categorical_feature"),
            callbacks=kwargs.get("callbacks"),
            init_model=kwargs.get("init_model"),
        )
    
    def predict(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict(
            X=x,
            raw_score=kwargs.get("raw_score", False),
            start_iteration=kwargs.get("start_iteration", 0),
            num_iteration=kwargs.get("num_iteration", None),
            pred_leaf=kwargs.get("pred_leaf", False),
            pred_contrib=kwargs.get("pred_contrib", False),
            validate_features=kwargs.get("validate_features", False),
        )
    
    def predict_proba(self, *args, **kwargs):
        x = kwargs.get("X")
        if x is None:
            raise ValueError("X is None")
        
        return self.model.predict_proba(
            X=x,
            raw_score=kwargs.get("raw_score", False),
            start_iteration=kwargs.get("start_iteration", 0),
            num_iteration=kwargs.get("num_iteration", None),
            pred_leaf=kwargs.get("pred_leaf", False),
            pred_contrib=kwargs.get("pred_contrib", False),
            validate_features=kwargs.get("validate_features", False),
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
            "num_iterations": hparams.get("num_iterations", self.config.num_iterations),
            "num_leaves": hparams.get("num_leaves", self.config.num_leaves),
            "learning_rate": hparams.get("learning_rate", self.config.learning_rate),
            "tree_learner": hparams.get("tree_learner", self.config.tree_learner),
            "num_threads": hparams.get("num_threads", self.config.num_threads),
            "device": hparams.get("device", self.config.device),
            "random_seed": hparams.get("random_seed", self.config.random_seed),
            "deterministic": hparams.get("deterministic", self.config.deterministic),
            "early_stopping_round": hparams.get("early_stopping_round", self.config.early_stopping_round),
            "force_col_wise": hparams.get("force_col_wise", self.config.force_col_wise),
            "force_row_wise": hparams.get("force_row_wise", self.config.force_row_wise),
            "histogram_pool_size": hparams.get("histogram_pool_size", self.config.histogram_pool_size),
            "max_depth": hparams.get("max_depth", self.config.max_depth),
            "min_data_in_leaf": hparams.get("min_data_in_leaf", self.config.min_data_in_leaf),
            "min_sum_hessian_in_leaf": hparams.get("min_sum_hessian_in_leaf", self.config.min_sum_hessian_in_leaf),
            "bagging_fraction": hparams.get("bagging_fraction", self.config.bagging_fraction),
            "pos_bagging_fraction": hparams.get("pos_bagging_fraction", self.config.pos_bagging_fraction),
            "neg_bagging_fraction": hparams.get("neg_bagging_fraction", self.config.neg_bagging_fraction),
            "bagging_freq": hparams.get("bagging_freq", self.config.bagging_freq),
            "bagging_seed": hparams.get("bagging_seed", self.config.bagging_seed),
            "feature_fraction": hparams.get("feature_fraction", self.config.feature_fraction),
            "feature_fraction_bynode": hparams.get("feature_fraction_bynode", self.config.feature_fraction_bynode),
            "feature_fraction_seed": hparams.get("feature_fraction_seed", self.config.feature_fraction_seed),
            "extra_trees": hparams.get("extra_trees", self.config.extra_trees),
            "extra_trees": hparams.get("extra_trees", self.config.extra_trees),
            "first_metric_only": hparams.get("first_metric_only", self.config.first_metric_only),
            "max_delta_step": hparams.get("max_delta_step", self.config.max_delta_step),
            "lambda_l1": hparams.get("lambda_l1", self.config.lambda_l1),
            "lambda_l2": hparams.get("lambda_l2", self.config.lambda_l2),
            "linear_lambda": hparams.get("linear_lambda", self.config.linear_lambda),
            "min_gain_to_split": hparams.get("min_gain_to_split", self.config.min_gain_to_split),
            "drop_rate": hparams.get("drop_rate", self.config.drop_rate),
            "max_drop": hparams.get("max_drop", self.config.max_drop),
            "skip_drop": hparams.get("skip_drop", self.config.skip_drop),
            "xgboost_dart_mode": hparams.get("xgboost_dart_mode", self.config.xgboost_dart_mode),
            "uniform_drop": hparams.get("uniform_drop", self.config.uniform_drop),
            "drop_seed": hparams.get("drop_seed", self.config.drop_seed),
            "top_rate": hparams.get("top_rate", self.config.top_rate),
            "other_rate": hparams.get("other_rate", self.config.other_rate),
            "min_data_per_group": hparams.get("min_data_per_group", self.config.min_data_per_group),
            "max_cat_threshold": hparams.get("max_cat_threshold", self.config.max_cat_threshold),
            "cat_l2": hparams.get("cat_l2", self.config.cat_l2),
            "cat_smooth": hparams.get("cat_smooth", self.config.cat_smooth),
            "max_cat_to_onehot": hparams.get("max_cat_to_onehot", self.config.max_cat_to_onehot),
            "top_k": hparams.get("top_k", self.config.top_k),
            "monotone_constraints": hparams.get("monotone_constraints", self.config.monotone_constraints),
            "monotone_constraints_method": hparams.get("monotone_constraints_method", self.config.monotone_constraints_method),
            "monotone_penalty": hparams.get("monotone_penalty", self.config.monotone_penalty),
            "feature_contri": hparams.get("feature_contri", self.config.feature_contri),
            "forcedsplits_filename": hparams.get("forcedsplits_filename", self.config.forcedsplits_filename),
            "interaction_constraints": hparams.get("interaction_constraints", self.config.interaction_constraints),
            "refit_decay_rate": hparams.get("refit_decay_rate", self.config.refit_decay_rate),
            "cegb_tradeoff": hparams.get("cegb_tradeoff", self.config.cegb_tradeoff),
            "cegb_penalty_split": hparams.get("cegb_penalty_split", self.config.cegb_penalty_split),
            "output_model": hparams.get("output_model", self.config.output_model),
            "saved_feature_importance_type": hparams.get("saved_feature_importance_type", self.config.saved_feature_importance_type),
            "path_smooth": hparams.get("path_smooth", self.config.path_smooth),
            "verbosity": hparams.get("verbosity", self.config.verbosity),
            "snapshot_freq": hparams.get("snapshot_freq", self.config.snapshot_freq),
            "use_quantized_grad": hparams.get("use_quantized_grad", self.config.use_quantized_grad),
            "num_grad_quant_bins": hparams.get("num_grad_quant_bins", self.config.num_grad_quant_bins),
            "quant_train_renew_leaf": hparams.get("quant_train_renew_leaf", self.config.quant_train_renew_leaf),
            "stochastic_rounding": hparams.get("stochastic_rounding", self.config.stochastic_rounding),
        }
        
        if isinstance(model, LGBMClassifier):
            model = LGBMClassifier(**config)
        elif isinstance(model, LGBMRegressor):
            model = LGBMRegressor(**config)
        
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