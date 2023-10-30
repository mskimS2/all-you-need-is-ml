import argparse


def svc_config():
    p = argparse.ArgumentParser(description="svc")
    
    p.add_argument("--model_name", type=str, default="svc")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--use_predict_proba", type=bool, default=True)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--problem_type", type=str, default="binary_classification")
    p.add_argument("--train_data", type=str, default="dataset/binary_classification.csv")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--fast", type=bool, default=True)
    p.add_argument("--output_path", type=str, default="results")
    
    # parameters 
    # - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    p.add_argument("--C", type=float, default=1.)
    p.add_argument("--kernel", type=str, default="rbf", choices=["linear", "poly", "rbf", "sigmoid", "precomputed"])
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--gamma", type=str, default="scale", choices=["scale", "auto"])
    p.add_argument("--coef0", type=float, default=0.)
    p.add_argument("--shrinking", type=bool, default=False)
    p.add_argument("--probability", type=bool, default=True)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--cache_size", type=float, default=200)
    p.add_argument("--class_weight", type=str, default=None, choices=["balanced", "None"])
    p.add_argument("--verbose", type=bool, default=False, choices=[True, False])
    p.add_argument("--max_iter", type=int, default=-1)
    p.add_argument("--decision_function_shape", type=str, default="ovr", choices=["ovo", "ovr"])
    p.add_argument("--break_ties", type=bool, default=False, choices=[True, False])
    p.add_argument("--random_state", type=int, default=42)
    
    return p.parse_args()

def svr_config():
    p = argparse.ArgumentParser(description="svr")
    
    p.add_argument("--model_name", type=str, default="svr")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--use_predict_proba", type=bool, default=True)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--verbose", type=bool, default=False)
    p.add_argument("--problem_type", type=str, default="binary_classification")
    p.add_argument("--train_data", type=str, default="dataset/binary_classification.csv")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--fast", type=bool, default=True)
    p.add_argument("--output_path", type=str, default="results")
    
    # parameters 
    # - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    p.add_argument("--C", type=float, default=1.)
    p.add_argument("--kernel", type=str, default="rbf", choices=["linear", "poly", "rbf", "sigmoid", "precomputed"])
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--gamma", type=str, default="scale", choices=["scale", "auto"])
    p.add_argument("--coef0", type=float, default=0.)
    p.add_argument("--shrinking", type=bool, default=False)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--cache_size", type=float, default=200)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--verbose", type=bool, default=False, choices=[True, False])
    p.add_argument("--max_iter", type=int, default=-1)
    
    return p.parse_args()