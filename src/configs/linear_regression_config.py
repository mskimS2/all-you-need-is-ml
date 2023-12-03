import argparse


def linear_regression_regressor_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="linear_regression")
    
    p.add_argument("--model_name", type=str, default="linear_regression")
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
    # - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
    p.add_argument("--fit_intercept", type=bool, default=True, choices=[True, False])
    p.add_argument("--copy_X", type=bool, default=True, choices=[True, False])
    p.add_argument("--n_job", type=int, default=None)
    p.add_argument("--positive", type=bool, default=False, choices=[True, False])
    
    args, _ = p.parse_known_args(args=[])
    return args