import copy
import numpy as np
from .type import Problem
from sklearn import metrics
from dataclasses import dataclass
from functools import partial

@dataclass
class BinaryClassificationMetric:
    metrics = {
        "auc": metrics.roc_auc_score,
        "logloss": metrics.log_loss,
        "f1_score": metrics.f1_score,
        "accuracy": metrics.accuracy_score,
        "precision": metrics.precision_score,
        "recall": metrics.recall_score,
    }

@dataclass
class MultiClassClassificationMetric:
    metrics = {
        "logloss": metrics.log_loss,
        "accuracy": metrics.accuracy_score,
        "mlogloss": metrics.log_loss,
    }    

@dataclass
class RegressionMetric:
    metrics = {
        "r2": metrics.r2_score,
        "mse": metrics.mean_squared_error,
        "mae": metrics.mean_absolute_error,
        "rmse": partial(metrics.mean_squared_error, squared=False),
        "rmsle": partial(metrics.mean_squared_log_error, squared=False),
    } 

@dataclass
class MultiLabelClassificationMetric:
    metrics = {
        "logloss": metrics.log_loss,
    } 

@dataclass
class Metric:
    problem: Problem
    metric = {
        problem["binary_classification"]: BinaryClassificationMetric(),
        problem["multi_class_classification"]: MultiLabelClassificationMetric(),
        problem["regression"]: RegressionMetric(),
        problem["multi_label_classification"]: MultiLabelClassificationMetric(),
    }
    
    def __post_init__(self):
        if self.problem in Problem.type.values():
            self._metric = Metric.metric[self.problem]
    
        raise ValueError(f"Invalid problem type...")
    
    def calculate(self, y_true, y_pred):
        results = {}
        for metric_name, metric_func in self._metric.metrics.items():
            if self.problem_type == Problem.binary_classification:
                if metric_name == "auc":
                    metrics[metric_name] = metric_func(y_true, y_pred[:, 1])
                elif metric_name == "logloss":
                    metrics[metric_name] = metric_func(y_true, y_pred)
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred[:, 1] >= 0.5)
            elif self.problem_type == Problem.multi_class_classification:
                if metric_name == "accuracy":
                    metrics[metric_name] = metric_func(y_true, np.argmax(y_pred, axis=1))
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred)
            else:
                if metric_name == "rmsle":
                    temp_pred = copy.deepcopy(y_pred)
                    temp_pred[temp_pred < 0] = 0
                    metrics[metric_name] = metric_func(y_true, temp_pred)
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred)
        return results

if __name__ == "__main__":
    pass
