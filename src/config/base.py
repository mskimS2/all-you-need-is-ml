from dataclasses import dataclass
from typing import Dict, List


@dataclass
class BaseConfig:
    model_name: str = "catboost"
    num_folds: int = 5
    random_seed: int = 42
    use_predict_proba: bool = True
    shuffle: bool = True
    verbose: bool = False
    problem_type: str = "binary_classification"
    train_data: str = "dataset/binary_classification.csv"
    only_one_train: bool = True
    output_path: str = "results"