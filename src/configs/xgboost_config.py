import argparse


def xgboost_classifier_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost_classifier")
    
    p.add_argument("--model_name", type=str, default="xgboost")
    p.add_argument("--task", type=str, default="classification")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--use_predict_proba", type=bool, default=True)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--problem_type", type=str, default="binary_classification")
    p.add_argument("--train_data", type=str, default="dataset/binary_classification.csv")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--only_one_train", type=bool, default=True)
    p.add_argument("--output_path", type=str, default="results")
    
    # parameters
    # - https://xgboost.readthedocs.io/en/stable/parameter.html
    p.add_argument("--learning_rate", type=float, default=0.3, help="eta(Learning rate), [0, 1]")
    p.add_argument("--gamma", type=float, default=0., help="gamma(min_split_loss), [0, inf]")
    p.add_argument("--max_depth", type=int, default=6, help="max_depth, [0, inf]")
    p.add_argument("--max_child_weight", type=int, default=1, help="max_child_weight, [0, inf]")
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--max_delta_step", type=int, default=0, help="max_delta_step, [0, inf]")
    p.add_argument("--subsample", type=float, default=1., help="max_delta_step, (0, 1]")
    p.add_argument("--sampling_method", type=str, default="uniform", help="(uniform, gradient_based)")
    p.add_argument("--colsample_bytree", type=float, default=1, help="colsample_bytree, (0, 1]")
    p.add_argument("--reg_lambda", type=float, default=1., help="lambda(reg_lambda), [0, inf]")
    p.add_argument("--alpha", type=float, default=0., help="alpha, [0, inf]")
    p.add_argument("--tree_method", type=str, default="auto", help="(auto, exact, approx, hist)")
    p.add_argument("--scale_pos_weight", type=float, default=1., help="A typical value to consider: sum(negative instances) / sum(positive instances).")
    p.add_argument("--grow_policy", type=str, default="depthwise", help="(depthwise, lossguide)")
    p.add_argument("--max_leaves", type=int, default=0, help="Maximum number of nodes to be added. Not used by exact tree method.")
    
    args, _ = p.parse_known_args(args=[])
    return args

def xgboost_regressor_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="XGBoost_regressor")
    
    p.add_argument("--model_name", type=str, default="xgboost")
    p.add_argument("--task", type=str, default="regression")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--problem_type", type=str, default="binary_classification")
    p.add_argument("--train_data", type=str, default="dataset/binary_classification.csv")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--only_one_train", type=bool, default=True)
    p.add_argument("--output_path", type=str, default="results")
    
    # parameters
    # - https://xgboost.readthedocs.io/en/stable/parameter.html
    p.add_argument("--learning_rate", type=float, default=0.3, help="eta(Learning rate), [0, 1]")
    p.add_argument("--gamma", type=float, default=0., help="gamma(min_split_loss), [0, inf]")
    p.add_argument("--max_depth", type=int, default=6, help="max_depth, [0, inf]")
    p.add_argument("--max_child_weight", type=int, default=1, help="max_child_weight, [0, inf]")
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--max_delta_step", type=int, default=0, help="max_delta_step, [0, inf]")
    p.add_argument("--subsample", type=float, default=1., help="max_delta_step, (0, 1]")
    p.add_argument("--sampling_method", type=str, default="uniform", help="(uniform, gradient_based)")
    p.add_argument("--colsample_bytree", type=float, default=1, help="colsample_bytree, (0, 1]")
    p.add_argument("--reg_lambda", type=float, default=1., help="lambda(reg_lambda), [0, inf]")
    p.add_argument("--alpha", type=float, default=0., help="alpha, [0, inf]")
    p.add_argument("--tree_method", type=str, default="auto", help="(auto, exact, approx, hist)")
    p.add_argument("--scale_pos_weight", type=float, default=1., help="A typical value to consider: sum(negative instances) / sum(positive instances).")
    p.add_argument("--grow_policy", type=str, default="depthwise", help="(depthwise, lossguide)")
    p.add_argument("--max_leaves", type=int, default=0, help="Maximum number of nodes to be added. Not used by exact tree method.")
    
    args, _ = p.parse_known_args(args=[])
    return args