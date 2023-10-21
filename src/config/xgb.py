import argparse


def xgboost_args():
    p = argparse.ArgumentParser(description="XGBoost")
    
    p.add_argument('--model_name', type=str, default="xgboost")
    
    return p.parse_args()