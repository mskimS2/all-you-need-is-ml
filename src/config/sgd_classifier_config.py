import argparse


def sgd_classifier_config() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sgd_classifier")
    
    p.add_argument("--model_name", type=str, default="sgd_classifier")
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
    # - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    p.add_argument("--loss", type=str, default="hinge", choices=["hinge", "log", "modified_huber", "squared_hinge", "perceptron", "squared_loss", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"])
    p.add_argument("--penalty", type=str, default="l2", choices=["l2", "l1", "elasticnet", "None"])
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--copy_X", type=bool, default=True)
    p.add_argument("--l1_ratio", type=float, default=0.15)
    p.add_argument("--fit_intercept", type=bool, default=False, choices=[True, False])
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--tol", type=bool, default=float, choices=1e-3)
    p.add_argument("--shuffle", type=bool, default=True)
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--n_jobs", type=int, default=None)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--learning_rate", type=str, default="optimal")
    p.add_argument("--eta0", type=float, default=0.)
    p.add_argument("--power_t", type=float, default=0.5)
    p.add_argument("--early_stopping", type=bool, default=False)
    p.add_argument("--validation_fraction", type=float, default=0.1)
    p.add_argument("--n_iter_no_change", type=int, default=5)
    p.add_argument("--class_weight", type=str, default=None, choices=["balanced", "None"])
    p.add_argument("--warm_start", type=bool, default=False)
    p.add_argument("--average", type=bool, default=False)
    
    return p.parse_args()