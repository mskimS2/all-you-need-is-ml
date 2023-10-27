import argparse


def xgboost_args():
    p = argparse.ArgumentParser(description="XGBoost")
    
    p.add_argument('--model_name', type=str, default="xgboost")
    p.add_argument('--num_folds', type=int, default=5)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--use_predict_proba', type=bool, default=True)
    p.add_argument('--shuffle', type=bool, default=True)
    p.add_argument('--early_stopping_rounds', type=int, default=50)
    p.add_argument('--problem_type', type=str, default="binary_classification")
    p.add_argument('--train_data', type=str, default="dataset/binary_classification.csv")
    p.add_argument('--device', type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument('--fast', type=bool, default=True)
    
    # parameters for tree booster
    # - https://xgboost.readthedocs.io/en/stable/parameter.html
    p.add_argument('--learning_rate', type=float, default=0.3, help="eta(Learning rate), [0, 1]")
    p.add_argument('--gamma', type=float, default=0, help="gamma(min_split_loss), [0, inf]")
    p.add_argument('--max_depth', type=int, default=6, help="max_depth, [0, inf]")
    p.add_argument('--max_child_weight', type=int, default=1, help="max_child_weight, [0, inf]")
    
    
    return p.parse_args()