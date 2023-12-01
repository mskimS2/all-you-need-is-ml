import argparse


def knn_classifier_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="knn_classifier")
    
    p.add_argument("--model_name", type=str, default="knn_classifier")
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
    # - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    p.add_argument("--n_neighbors", type=int, default=5)
    p.add_argument("--weights", type=str, default="uniform", choices=["uniform", "distance"])
    p.add_argument("--algorithm", type=str, default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"])
    p.add_argument("--gamma", type=str, default="scale", choices=["scale", "auto"])
    p.add_argument("--leaf_size", type=int, default=30)
    p.add_argument("--p", type=float, default=2)
    p.add_argument("--metric", type=str, default="minkowski")
    p.add_argument("--metric_params", type=str, default=None)
    p.add_argument("--n_jobs", type=int, default=None)
    
    args, _ = p.parse_known_args(args=[])
    return args

def knn_regressor_config():
    p = argparse.ArgumentParser(description="knn_regressor")
    
    p.add_argument("--model_name", type=str, default="knn_regressor")
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
    # - https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    p.add_argument("--n_neighbors", type=int, default=5)
    p.add_argument("--weights", type=str, default="uniform", choices=["uniform", "distance"])
    p.add_argument("--algorithm", type=str, default="auto", choices=["auto", "ball_tree", "kd_tree", "brute"])
    p.add_argument("--gamma", type=str, default="scale", choices=["scale", "auto"])
    p.add_argument("--leaf_size", type=int, default=30)
    p.add_argument("--p", type=float, default=2)
    p.add_argument("--metric", type=str, default="minkowski")
    p.add_argument("--metric_params", type=str, default=None)
    p.add_argument("--n_jobs", type=int, default=None)
    
    args, _ = p.parse_known_args(args=[])
    return args