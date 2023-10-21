import os
import optuna
import pandas as pd
import numpy as np
import xgboost
from typing import List
from src.logger import logger
from src.models import get_model
from src.metrics import Metrics

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def fetch_model(
    model_name: str,
    problem: str,
    optimize: str, # minimize or maximize
):
    return {
        "model_name": model_name,
        "problem": problem,
        "optimize": optimize,
        "model": get_model(model_name, problem),
        "use_predict_proba": True
    }

def optimize(
    model_config,
    features: List[str],
    targets: List[str],
):
    metrics = Metrics(model_config.problem_type)
    eval_metric = metrics.metrics.keys()
    scores = []
    
    for fold in range(model_config.num_folds):
        train_feather = pd.read_feather(os.path.join(model_config.output, f"train_fold_{fold}.feather"))
        valid_feather = pd.read_feather(os.path.join(model_config.output, f"valid_fold_{fold}.feather"))
        x_train, y_train = train_feather[features], train_feather[targets].values
        x_valid, y_valid = valid_feather[features], valid_feather[targets].values

        # train model
        model = xgboost.XGBClassifier(
            random_state=model_config.random_seed,
            eval_metric=["error"],
            use_label_encoder=False,
            # **params,
        )

        ypred = []
        models = [model] * len(model_config.targets)
        for idx, _m in enumerate(models):
            _m.fit(
                x_train,
                y_train[:, idx],
                early_stopping_rounds=model_config.early_stopping_rounds,
                eval_set=[(x_valid, y_valid[:, idx])],
                verbose=False,
            )
           
            ypred_temp = _m.predict_proba(x_valid)[:, 1]
            ypred.append(ypred_temp)
        ypred = np.column_stack(ypred)

        if model_config.use_predict_proba:
            ypred = model.predict_proba(x_valid)
        else:
            ypred = model.predict(x_valid)

        # calculate metric
        metric_dict = metrics.calculate(y_valid, ypred)
        scores.append(metric_dict)
        if model_config.fast is True:
            break

    mean_metrics = dict_mean(scores)
    logger.info(f"Metrics: {mean_metrics}")
    return mean_metrics[eval_metric]


if __name__ == "__main__":
    from src.config.xgboost import xgboost_args
    args = xgboost_args()
    optimize(
        args,
        features=[],
        targets=["income"],
    )