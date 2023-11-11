from typing import Tuple
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.svm import (
    SVC, 
    SVR,
)
from sklearn.tree import (
    DecisionTreeClassifier, 
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.linear_model import (
    LogisticRegression, 
    LinearRegression,
    SGDClassifier,
    Lasso,
)
from sklearn.ensemble import (
    RandomForestClassifier,  
    RandomForestRegressor,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)

import argparse
from const import Const
from models.base import BaseModel
from models.xgboost import XGBoost
from config.xgboost_config import xgboost_args
from config.catboost_config import catboost_args
from config.lightgbm_config import lightgbm_args
from config.random_forest_config import random_forest_config
from config.extra_tree_config import extra_tree_config
from config.decision_tree_config import decision_tree_config
from config.logistic_regression_config import logistic_regression_config
from config.linear_regression_config import linear_regression_config
from config.lasso_config import lasso_config
from config.sgd_classifier_config import sgd_classifier_config
from config.support_vector_machine_config import (
    svc_config, 
    svr_config,
)
from config.knn_config import (
    knn_classifier_config,
    knn_regressor_config,
)


def get_xgboost(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = xgboost_args()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        model = XGBoost(XGBClassifier(), args) 
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = XGBoost(XGBRegressor(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    
    return model, args

def get_lightgbm(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = lightgbm_args()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        model = LGBMClassifier()
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = LGBMRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_catboost(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = catboost_args()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        model = CatBoostClassifier()
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = CatBoostRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_random_forest(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = random_forest_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        model = RandomForestClassifier()
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = RandomForestRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_decision_tree(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = decision_tree_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        model = DecisionTreeClassifier()
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = DecisionTreeRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_extra_tree(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = extra_tree_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        model = ExtraTreeClassifier()
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = ExtraTreeRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_logistic_regression(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = logistic_regression_config()
    if problem == "single_column_regression":
        model = LogisticRegression()
    elif problem == "multi_column_regression":
        model = LogisticRegression()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_linear_regression(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = linear_regression_config()
    if problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = LinearRegression()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_lasso(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = lasso_config()
    if problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = Lasso()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_sgd_classifier(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    args = sgd_classifier_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        model = SGDClassifier()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_support_vector_machine(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args = svc_config()
        model = SVC()
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args = svr_config()
        model = SVR()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_knn(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args = knn_classifier_config()
        model = KNeighborsClassifier()
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args = knn_regressor_config()
        model = KNeighborsRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_model(model_name: str, problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    model_dict = {
        Const.XGBOOST: get_xgboost,
        Const.CATBOOST: get_catboost,
        Const.LIGHTGBM: get_lightgbm,
        Const.RANDOM_FOREST: get_random_forest,
        Const.DECISION_TREE: get_decision_tree,
        Const.EXTRA_TREE: get_extra_tree,
        Const.LOGISTIC_REGRESSION: get_logistic_regression,
        Const.LINEAR_REGRESSION: get_linear_regression,
        Const.LASSO: get_lasso,
        Const.SGD_CLASSIFIER: get_sgd_classifier,
        Const.SVM: get_support_vector_machine,
        Const.KNN: get_knn,
    }
    
    if model_dict.get(model_name) is None:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return model_dict[model_name](problem)