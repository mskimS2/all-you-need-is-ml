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
    RandomForestRegressor
)
from config.xgboost_config import xgboost_args
from config.catboost_config import catboost_args
from config.lightgbm_config import lightgbm_args
from config.random_forest_config import random_forest_config
from config.extra_tree_config import extra_tree_config
from config.decision_tree_config import decision_tree_config
from config.logistic_regression_config import logistic_regression_config


def get_xgboost(problem: str):
    args = xgboost_args()
    if problem == "binary_classification":
        model = XGBClassifier()
    elif problem == "multi_class_classification":
        model = XGBClassifier()
    elif problem == "multi_label_classification":
        model = XGBClassifier()
    elif problem == "single_column_regression":
        model = XGBRegressor()
    elif problem == "multi_column_regression":
        model = XGBRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_lightgbm(problem: str):
    args = lightgbm_args()
    if problem == "binary_classification":
        model = LGBMClassifier()
    elif problem == "multi_class_classification":
        model = LGBMClassifier()
    elif problem == "multi_label_classification":
        model = LGBMClassifier()
    elif problem == "single_column_regression":
        model = LGBMRegressor()
    elif problem == "multi_column_regression":
        model = LGBMRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_catboost(problem: str):
    args = catboost_args()
    if problem == "binary_classification":
        model = CatBoostClassifier()
    elif problem == "multi_class_classification":
        model = CatBoostClassifier()
    elif problem == "multi_label_classification":
        model = CatBoostClassifier()
    elif problem == "single_column_regression":
        model = CatBoostRegressor()
    elif problem == "multi_column_regression":
        model = CatBoostRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_random_forest(problem: str):
    args = random_forest_config()
    if problem == "binary_classification":
        model = RandomForestClassifier()
    elif problem == "multi_class_classification":
        model = RandomForestClassifier()
    elif problem == "multi_label_classification":
        model = RandomForestClassifier()
    elif problem == "single_column_regression":
        model = RandomForestRegressor()
    elif problem == "multi_column_regression":
        model = RandomForestRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_decision_tree(problem: str):
    args = decision_tree_config()
    if problem == "binary_classification":
        model = DecisionTreeClassifier()
    elif problem == "multi_class_classification":
        model = DecisionTreeClassifier()
    elif problem == "multi_label_classification":
        model = DecisionTreeClassifier()
    elif problem == "single_column_regression":
        model = DecisionTreeRegressor()
    elif problem == "multi_column_regression":
        model = DecisionTreeRegressor()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_extra_tree(problem: str):
    args = extra_tree_config()
    if problem == "binary_classification":
        model = ExtraTreeClassifier()
    elif problem == "multi_class_classification":
        model = ExtraTreeClassifier()
    elif problem == "multi_label_classification":
        model = ExtraTreeClassifier()
    elif problem == "single_column_regression":
        model = ExtraTreeRegressor()
    elif problem == "multi_column_regression":
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
    args = None
    if problem == "single_column_regression":
        model = LinearRegression()
    elif problem == "multi_column_regression":
        model = LinearRegression()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_lasso(problem: str):
    args = None
    if problem == "single_column_regression":
        model = Lasso()
    elif problem == "multi_column_regression":
        model = Lasso()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_sgd_classifier(problem: str):
    args = None
    if problem == "binary_classification":
        model = SGDClassifier()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_support_vector_machine(problem: str):
    args = None
    if problem == "binary_classification":
        model = SVC()
    elif problem == "multi_class_classification":
        model = SVC()
    elif problem == "multi_label_classification":
        model = SVC()
    elif problem == "single_column_regression":
        model = SVR()
    elif problem == "multi_column_regression":
        model = SVR()
    else:
        raise ValueError(f"Invalid problem type: {problem}")
    return model, args

def get_model(
    model_name: str,
    problem: str,
):
    if model_name == "xgboost":
        return get_xgboost(problem)
    elif model_name == "catboost":
        return get_catboost(problem)
    elif model_name == "lightgbm":
        return get_lightgbm(problem)
    elif model_name == "random_forest":
        return get_random_forest(problem)
    elif model_name == "decision_tree":
        return get_decision_tree(problem)
    elif model_name == "extra_tree":
        return get_extra_tree(problem)
    elif model_name == "logistic_regression":
        return get_logistic_regression(problem)
    elif model_name == "linear_regression":
        return get_linear_regression(problem)
    elif model_name == "lasso":
        return get_lasso(problem)
    elif model_name == "sgd_classifier":
        return get_sgd_classifier(problem)
    elif model_name == "support_vector_machine":
        return get_support_vector_machine(problem)
    else:
        raise ValueError(f"Invalid model name: {model_name}")