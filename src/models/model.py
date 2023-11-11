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

from const import Const
from models.xgboost import XGBoost


def get_xgboost(problem: str):
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

def get_lightgbm(problem: str):
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

def get_catboost(problem: str):
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

def get_random_forest(problem: str):
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

def get_decision_tree(problem: str):
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

def get_extra_tree(problem: str):
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

def get_logistic_regression(problem: str):
    args = logistic_regression_config()
    if problem == "single_column_regression":
        model = LogisticRegression()
    elif problem == "multi_column_regression":
        model = LogisticRegression()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_linear_regression(problem: str):
    args = linear_regression_config()
    if problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = LinearRegression()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_lasso(problem: str):
    args = lasso_config()
    if problem in [
        Const.SINGLE_COLUMN_REGRESSION,
        Const.MULTI_COLUMN_REGRESSION,
    ]:
        model = Lasso()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_sgd_classifier(problem: str):
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

def get_support_vector_machine(problem: str):
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

def get_knn(problem: str):
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

def get_model(
    model_name: str,
    problem: str,
):
    model_dict = {
        Const.XGBOOST: get_xgboost(problem),
        Const.CATBOOST: get_catboost(problem),
        Const.LIGHTGBM: get_lightgbm(problem),
        Const.RANDOM_FOREST: get_random_forest(problem),
        Const.DECISION_TREE: get_decision_tree(problem),
        Const.EXTRA_TREE: get_extra_tree(problem),
        Const.LOGISTIC_REGRESSION: get_logistic_regression(problem),
        Const.LINEAR_REGRESSION: get_linear_regression(problem),
        Const.LASSO: get_lasso(problem),
        Const.SGD_CLASSIFIER: get_sgd_classifier(problem),
        Const.SVM: get_support_vector_machine(problem),
        Const.KNN: get_knn(problem),
    }
    
    if model_dict.get(model_name) is None:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return model_dict[model_name]