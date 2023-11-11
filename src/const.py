from typing import Final
from dataclasses import dataclass


@dataclass
class Const:
    CLASSIFICATION: Final = "classification"
    REGRESSION: Final = "regression"
    BINARY_CLASSIFICATION: Final = "binary_classification"
    MULTI_CLASS_CLASSIFICATION: Final = "multi_class_classification"
    MULTI_LABEL_CLASSIFICATION: Final = "multi_label_classification"
    SINGLE_COLUMN_REGRESSION: Final = "single_column_regression"
    MULTI_COLUMN_REGRESSION: Final = "multi_column_regression"
    
    # metric
    AUC: Final = "auc"
    LOG_LOSS: Final = "logloss"
    F1_SCORE: Final = "f1_score"
    ACCURACY: Final = "accuracy"
    PRECISION: Final = "precision"
    RECALL: Final = "recall"
    R2: Final = "r2"
    MSE: Final = "mse"
    MAE: Final = "mae"
    RMSE: Final = "rmse"
    RMSLE: Final = "rmsle"
    MLOG_LOSS: Final = "mlogloss"
    
    # model
    XGBOOST: Final = "xgboost"
    CATBOOST: Final = "catboost"
    LIGHTGBM: Final = "lightgbm"
    RANDOM_FOREST: Final = "random_forest"
    DECISION_TREE: Final = "decision_tree"
    EXTRA_TREE: Final = "extra_tree"
    LOGISTIC_REGRESSION: Final = "logistic_regression"
    LINEAR_REGRESSION: Final = "linear_regression"
    LASSO: Final = "lasso"
    SGD_CLASSIFIER: Final = "sgd_classifier"
    SVM: Final = "support_vector_machine"
    KNN: Final = "knn"
    
    # column name
    FOLD_ID: Final = "fold_id"