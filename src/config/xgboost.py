import argparse


def xgboost_args():
    p = argparse.ArgumentParser(description="XGBoost")
    
    p.add_argument('--model_name', type=str, default="xgboost")
    p.add_argument('--num_folds', type=int, default=5)
    p.add_argument('--random_seed', type=int, default=42)
    p.add_argument('--use_predict_proba', type=bool, default=True)
    p.add_argument('--early_stopping_rounds', type=int, default=50)
    p.add_argument('--problem_type', type=str, default="binary_classification")
    p.add_argument('--problem_type', type=str, default="dataset/binary_classification.csv")
    
    return p.parse_args()