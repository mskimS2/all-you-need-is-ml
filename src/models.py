import xgboost as xgb

def get_model(
    model_name: str,
    problem: str,
):
    model = None
    
    if model_name == "xgboost":
        if problem == "binary_classification":
            model = xgb.XGBClassifier
        elif problem == "multi_class_classification":
            model = xgb.XGBClassifier
        elif problem == "multi_label_classification":
            model = xgb.XGBClassifier
        elif problem == "single_column_regression":
            model = xgb.XGBRegressor
        elif problem == "multi_column_regression":
            model = xgb.XGBRegressor
        else:
            raise ValueError(f"Invalid problem type: {problem}")
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return model
    