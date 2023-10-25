from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn import svm
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

def get_model(
    model_name: str,
    problem: str,
):
    model = None
    
    if model_name == "xgboost":
        if problem == "binary_classification":
            model = XGBClassifier
        elif problem == "multi_class_classification":
            model = XGBClassifier
        elif problem == "multi_label_classification":
            model = XGBClassifier
        elif problem == "single_column_regression":
            model = XGBRegressor
        elif problem == "multi_column_regression":
            model = XGBRegressor
        else:
            raise ValueError(f"Invalid problem type: {problem}")
    
    elif model_name == "catboost":
        if problem == "binary_classification":
            model = CatBoostClassifier
        elif problem == "multi_class_classification":
            model = CatBoostClassifier
        elif problem == "multi_label_classification":
            model = CatBoostClassifier
        elif problem == "single_column_regression":
            model = CatBoostRegressor
        elif problem == "multi_column_regression":
            model = CatBoostRegressor
        else:
            raise ValueError(f"Invalid problem type: {problem}")
        
    elif model_name == "lightgbm":
        if problem == "binary_classification":
            model = LGBMClassifier
        elif problem == "multi_class_classification":
            model = LGBMClassifier
        elif problem == "multi_label_classification":
            model = LGBMClassifier
        elif problem == "single_column_regression":
            model = LGBMRegressor
        elif problem == "multi_column_regression":
            model = LGBMRegressor
        else:
            raise ValueError(f"Invalid problem type: {problem}")
        
    elif model_name == "random_forest":
        if problem == "binary_classification":
            model = RandomForestClassifier
        elif problem == "multi_class_classification":
            model = RandomForestClassifier
        elif problem == "multi_label_classification":
            model = RandomForestClassifier
        elif problem == "single_column_regression":
            model = RandomForestRegressor
        elif problem == "multi_column_regression":
            model = RandomForestRegressor
        else:
            raise ValueError(f"Invalid problem type: {problem}")
    
    elif model_name == "decision_tree":
        if problem == "binary_classification":
            model = DecisionTreeClassifier
        elif problem == "multi_class_classification":
            model = DecisionTreeClassifier
        elif problem == "multi_label_classification":
            model = DecisionTreeClassifier
        elif problem == "single_column_regression":
            model = DecisionTreeRegressor
        elif problem == "multi_column_regression":
            model = DecisionTreeRegressor
        else:
            raise ValueError(f"Invalid problem type: {problem}")
        
    elif model_name == "extra_tree":
        if problem == "binary_classification":
            model = ExtraTreeClassifier
        elif problem == "multi_class_classification":
            model = ExtraTreeClassifier
        elif problem == "multi_label_classification":
            model = ExtraTreeClassifier
        elif problem == "single_column_regression":
            model = ExtraTreeRegressor
        elif problem == "multi_column_regression":
            model = ExtraTreeRegressor
        else:
            raise ValueError(f"Invalid problem type: {problem}")

    elif model_name == "logistic_regression":
        if problem == "single_column_regression":
            model = LogisticRegression
        elif problem == "multi_column_regression":
            model = LogisticRegression
        else:
            raise ValueError(f"Invalid problem type: {problem}")
        
    elif model_name == "linear_regression":
        if problem == "single_column_regression":
            model = LinearRegression
        elif problem == "multi_column_regression":
            model = LinearRegression
        else:
            raise ValueError(f"Invalid problem type: {problem}")

    elif model_name == "lasso":
        if problem == "single_column_regression":
            model = Lasso
        elif problem == "multi_column_regression":
            model = Lasso
        else:
            raise ValueError(f"Invalid problem type: {problem}")
        
    elif model_name == "sgd_classifier":
        if problem == "binary_classification":
            model = SGDClassifier
        else:
            raise ValueError(f"Invalid problem type: {problem}")
        
    elif model_name == "support_vector_machine":
        if problem == "binary_classification":
            model = svm.SVC
        elif problem == "multi_class_classification":
            model = svm.SVC
        elif problem == "multi_label_classification":
            model = svm.SVC
        elif problem == "single_column_regression":
            model = svm.SVR
        elif problem == "multi_column_regression":
            model = svm.SVR
        else:
            raise ValueError(f"Invalid problem type: {problem}")
    
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return model