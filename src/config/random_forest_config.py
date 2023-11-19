import argparse


def random_forest_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="random_forest")
    
    p.add_argument("--model_name", type=str, default="random_forest")
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--use_predict_proba", type=bool, default=True)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--problem_type", type=str, default="binary_classification")
    p.add_argument("--train_data", type=str, default="dataset/binary_classification.csv")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    p.add_argument("--only_one_train", type=bool, default=True)
    p.add_argument("--output_path", type=str, default="results")
    
    # parameters 
    # - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    p.add_argument("--n_estimation", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=7)
    p.add_argument("--min_samples_split", type=int, default=2)
    p.add_argument("--min_samples_leaf", type=int, default=1)
    p.add_argument("--min_weight_fraction_leaf", type=float, default=0.)
    p.add_argument("--max_features", type=float, default=1.)
    p.add_argument("--max_leaf_nodes", type=float, default=None)
    p.add_argument("--min_impurity_decrease", type=float, default=0.)
    p.add_argument("--bootstrap", type=bool, default=True)
    p.add_argument("--oob_score", type=bool, default=False)
    p.add_argument("--n_jobs", type=int, default=None)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--warm_start", type=bool, default=False)
    p.add_argument("--ccp_alpha", type=float, default=0.)
    p.add_argument("--max_samples", type=float, default=None)
    
    args, _ = p.parse_known_args(args=[])
    return args