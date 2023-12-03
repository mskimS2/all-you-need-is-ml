import argparse


def lasso_regressor_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="lasso")
    
    p.add_argument("--model_name", type=str, default="lasso")
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
    # - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    p.add_argument("--alpha", type=float, default=1.)
    p.add_argument("--fit_intercept", type=bool, default=True, choices=[True, False])
    p.add_argument("--precompute", type=bool, default=False)
    p.add_argument("--copy_X", type=bool, default=True)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--warm_start", type=bool, default=False, choices=[True, False])
    p.add_argument("--positive", type=bool, default=False, choices=[True, False])
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--selection", type=str, default="cyclic", choices=["cyclic", "random"])
    
    args, _ = p.parse_known_args(args=[])
    return args