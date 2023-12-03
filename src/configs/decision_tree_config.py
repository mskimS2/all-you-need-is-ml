import argparse


def decision_tree_classifier_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="decision_tree_classifier")
    
    p.add_argument("--model_name", type=str, default="decision_tree")
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
    # - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    p.add_argument("--criterion", type=str, default="gini", choices=["gini", "entropy", "log_loss"])
    p.add_argument("--splitter", type=str, default="best", choices=["best", "random"])
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--min_weight_fraction_leaf", type=float, default=0.)
    p.add_argument("--max_features", type=float, default=None, choices=["auto", "sqrt", "log2"])
    p.add_argument("--max_leaf_nodes", type=float, default=None)
    p.add_argument("--min_impurity_decrease", type=float, default=0.)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--class_weight", type=float, default=None, choices=["balanced", "balanced_subsample"])
    p.add_argument("--ccp_alpha", type=float, default=0.)
    
    args, _ = p.parse_known_args(args=[])
    return args

def decision_tree_regressor_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="decision_tree_Regressor")
    
    p.add_argument("--model_name", type=str, default="decision_tree")
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
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
    p.add_argument("--criterion", type=str, default="squared_error", choices=["poisson", "absolute_error", "friedman_mse", "squared_error"])
    p.add_argument("--splitter", type=str, default="best", choices=["best", "random"])
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--min_weight_fraction_leaf", type=float, default=0.)
    p.add_argument("--max_features", type=float, default=None, choices=["auto", "sqrt", "log2"])
    p.add_argument("--max_leaf_nodes", type=float, default=None)
    p.add_argument("--min_impurity_decrease", type=float, default=0.)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--class_weight", type=float, default=None, choices=["balanced", "balanced_subsample"])
    p.add_argument("--ccp_alpha", type=float, default=0.)
    
    args, _ = p.parse_known_args(args=[])
    return args