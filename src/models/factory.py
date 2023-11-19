import argparse
from typing import Tuple

from const import Const
from models.base import BaseModel


def get_xgboost(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from xgboost import XGBClassifier, XGBRegressor
    from models.classification.xgboost import XGBoostClassifier
    from models.regressor.xgboost import XGBoostRegressor
    from config.xgboost_config import xgboost_args
    
    args = xgboost_args()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args.task = Const.CLASSIFICATION
        model = XGBoostClassifier(XGBClassifier(), args) 
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = XGBoostRegressor(XGBRegressor(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    
    return model, args

def get_lightgbm(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from lightgbm import LGBMClassifier, LGBMRegressor
    from models.classification.lightgbm import LightGBMClassifier
    from models.regressor.lightgbm import LightGBMRegressor
    from config.lightgbm_config import lightgbm_args
    
    args = lightgbm_args()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args.task = Const.CLASSIFICATION
        model = LightGBMClassifier(LGBMClassifier(), args)
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = LightGBMRegressor(LGBMRegressor(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_catboost(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from catboost import CatBoostClassifier, CatBoostRegressor
    from models.classification.catboost import CatBoostClassifier as CBC
    from models.regressor.catboost import CatBoostRegressor as CRC
    from config.catboost_config import catboost_args
    
    args = catboost_args()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args.task = Const.CLASSIFICATION
        model = CBC(CatBoostClassifier(), args)
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = CRC(CatBoostRegressor(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_random_forest(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from models.classification.random_forest import RandomForestClassifier as RFC
    from models.regressor.random_forest import RandomForestRegressor as RFR
    from config.random_forest_config import random_forest_config
    
    args = random_forest_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args.task = Const.CLASSIFICATION
        model = RFC(RandomForestClassifier(), args)
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = RFR(RandomForestRegressor(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_decision_tree(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from models.classification.decision_tree import DecisionTreeClassifier as DTC
    from models.regressor.decision_tree import DecisionTreeRegressor as DTR
    from config.decision_tree_config import decision_tree_config
    
    args = decision_tree_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args.task = Const.CLASSIFICATION
        model = DTC(DecisionTreeClassifier(), args)
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = DTR(DecisionTreeRegressor(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_extra_tree(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor
    from models.classification.extra_tree import ExtraTreeClassifier as ETC
    from models.regressor.extra_tree import ExtraTreeRegressor as ETR
    from config.extra_tree_config import extra_tree_config
    
    args = extra_tree_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args.task = Const.CLASSIFICATION
        model = ETC(ExtraTreeClassifier(), args)
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = ETR(ExtraTreeRegressor(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_logistic_regression(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.linear_model import LogisticRegression
    from config.logistic_regression_config import logistic_regression_config
    
    args = logistic_regression_config()
    if problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = LogisticRegression()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_linear_regression(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.linear_model import LinearRegression
    from config.linear_regression_config import linear_regression_config
    
    args = linear_regression_config()
    if problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = LinearRegression()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_lasso(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.linear_model import Lasso
    from config.lasso_config import lasso_config
    
    args = lasso_config()
    if problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args.task = Const.REGRESSION
        model = Lasso()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_sgd_classifier(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.linear_model import SGDClassifier
    from models.classification.sgd_classifier import SgdClassifier
    from config.sgd_classifier_config import sgd_classifier_config
    
    args = sgd_classifier_config()
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args.task = Const.CLASSIFICATION
        model = SgdClassifier(SGDClassifier(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_support_vector_machine(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.svm import SVC, SVR
    from models.classification.svm import SVMClassifier
    from models.regressor.svm import SVMRegressor
    from config.support_vector_machine_config import svc_config, svr_config
    
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args = svc_config()
        args.task = Const.CLASSIFICATION
        model = SVMClassifier(SVC(), args)
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args = svr_config()
        args.task = Const.REGRESSION
        model = SVMRegressor(SVR(), args)
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_knn(problem: str) -> Tuple[BaseModel, argparse.Namespace]:
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from models.regressor.knn import KNNRegressor
    from models.classification.knn import KNNClassifier
    from config.knn_config import knn_classifier_config, knn_regressor_config
    
    if problem in [
        Const.BINARY_CLASSIFICATION,
        Const.MULTI_CLASS_CLASSIFICATION,
        Const.MULTI_LABEL_CLASSIFICATION,
    ]:
        args = knn_classifier_config()
        args.task = Const.CLASSIFICATION
        model = KNNClassifier(KNeighborsClassifier(), args)
    elif problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        args = knn_regressor_config()
        args.task = Const.REGRESSION
        model = KNNRegressor(KNeighborsRegressor(), args)
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