import copy
import numpy as np
from type import Problem
from const import Const
from sklearn import metrics
from dataclasses import dataclass
from functools import partial

@dataclass
class BinaryClassificationMetric:
    metrics = {
        Const.AUC: metrics.roc_auc_score,
        Const.LOG_LOSS: metrics.log_loss,
        Const.F1_SCORE: metrics.f1_score,
        Const.ACCURACY: metrics.accuracy_score,
        Const.PRECISION: metrics.precision_score,
        Const.RECALL: metrics.recall_score,
    }

@dataclass
class MultiClassClassificationMetric:
    metrics = {
        Const.LOG_LOSS: metrics.log_loss,
        Const.ACCURACY: metrics.accuracy_score,
        Const.MLOG_LOSS: metrics.log_loss,
    }    

@dataclass
class RegressionMetric:
    metrics = {
        Const.R2: metrics.r2_score,
        Const.MSE: metrics.mean_squared_error,
        Const.MAE: metrics.mean_absolute_error,
        Const.RMSE: partial(metrics.mean_squared_error, squared=False),
        Const.RMSLE: partial(metrics.mean_squared_log_error, squared=False),
    } 

@dataclass
class MultiLabelClassificationMetric:
    metrics = {
        Const.LOG_LOSS: metrics.log_loss,
    } 

@dataclass
class Metric:
    problem: Problem
    metric = {
        Const.BINARY_CLASSIFICATION: BinaryClassificationMetric(),
        Const.MULTI_CLASS_CLASSIFICATION: MultiLabelClassificationMetric(),
        Const.SINGLE_COLUMN_REGRESSION: RegressionMetric(),
        Const.MULTI_LABEL_CLASSIFICATION: MultiLabelClassificationMetric(),
        Const.MULTI_COLUMN_REGRESSION: RegressionMetric(),
    }
    
    def __post_init__(self):
        self._metric = Metric.metric.get(self.problem)
        if self._metric is None:
            raise ValueError(f"Invalid problem type...")
    
    def calculate(self, y_true, y_pred):
        results = {}
        for metric_name, metric_func in self._metric.metrics.items():
            if self.problem == Const.BINARY_CLASSIFICATION:
                if metric_name == Const.AUC:
                    results[metric_name] = metric_func(y_true, y_pred[:, 1])
                elif metric_name == Const.LOG_LOSS:
                    results[metric_name] = metric_func(y_true, y_pred)
                else:
                    results[metric_name] = metric_func(y_true, y_pred[:, 1] >= 0.5)
            elif self.problem == Const.MULTI_CLASS_CLASSIFICATION:
                if metric_name == Const.ACCURACY:
                    results[metric_name] = metric_func(y_true, np.argmax(y_pred, axis=1))
                else:
                    results[metric_name] = metric_func(y_true, y_pred)
            else:
                if metric_name == Const.RMSLE:
                    temp_pred = copy.deepcopy(y_pred)
                    temp_pred[temp_pred < 0] = 0
                    results[metric_name] = metric_func(y_true, temp_pred)
                else:
                    results[metric_name] = metric_func(y_true, y_pred)
        return results
