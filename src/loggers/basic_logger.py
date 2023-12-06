import logging
from .logger import Logger


class BasicLogger(Logger):
    def init_logger(self):
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

        self.logger = logging.getLogger("logger")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    def __del__(self):
        try:
            self.finish_experiment()
        except Exception:
            pass

    def __repr__(self) -> str:
        return self.__class__.__name__

    def log_params(self, params, model_name=None):
        self.logger.info(f"Params: {params}, model_name: {model_name}")

    def log_metrics(self, metrics, source=None):
        self.logger.info(f"metrics: {metrics}, source: {source}")

    def finish_experiment(self):
        self.logger.info("Experiment finished")
    
