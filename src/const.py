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
    
    # column name
    FOLD_ID: Final = "fold_id"