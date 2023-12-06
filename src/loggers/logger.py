import logging
from abc import ABC


class Logger(ABC):
    def init_logger(self):
        pass

    def __del__(self):
        try:
            self.finish_experiment()
        except Exception:
            pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    def log_params(self, params, model_name=None):
        pass

    def log_metrics(self, metrics, source=None):
        pass

    def finish_experiment(self):
        pass