import argparse


def logistic_regression_regressor_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="logistic_regression")
    
    p.add_argument("--model_name", type=str, default="logistic_regression")
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
    # - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    p.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2", "elasticnet", "None"])
    p.add_argument("--dual", type=bool, default=False)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--C", type=float, default=1.)
    p.add_argument("--fit_intercept", type=bool, default=True)
    p.add_argument("--intercept_scaling", type=float, default=1.)
    p.add_argument("--class_weight", type=str, default=None, choices=["balanced", "None"])
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--solver", type=str, default="lbfgs", choices=["newton-cg", "newton-cholesky", "lbfgs", "liblinear", "sag", "saga"])
    
    args, _ = p.parse_known_args(args=[])
    return args