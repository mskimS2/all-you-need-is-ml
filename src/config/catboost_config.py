import argparse


def catboost_args():
    p = argparse.ArgumentParser(description="Catboost")
    
    p.add_argument("--model_name", type=str, default="catboost")
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
    # - https://catboost.ai/en/docs/references/training-parameters/
    p.add_argument("--iterations", type=float, default=1000, help="[1, inf]")
    p.add_argument("--learning_rate", type=float, default=0., help="learning_rate, (0, inf]")
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--cat_features", type=str, default="", help="cat_features, col1, col2...")
    
    return p.parse_args()