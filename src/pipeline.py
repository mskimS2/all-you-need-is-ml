from copy import deepcopy
from dataclasses import dataclass
from sklearn.pipeline import Pipeline, make_pipeline


def get_estimator_from_pipeline(pipeline: Pipeline):
    return pipeline._final_estimator

def add_estimator_to_pipeline(pipeline: Pipeline, estimator, name: str="estimator"):
    try:
        assert hasattr(pipeline._final_estimator, "predict")
        pipeline.replace_final_estimator(estimator, name=name)
    except Exception:
        pipeline.steps.append((name, estimator))

class transform_pipeline:
    def __init__(self, pipeline, estimator, config):
        self.pipeline = deepcopy(pipeline)
        self.estimator = estimator(**config)
        
    def __enter__(self):
        if isinstance(self.estimator, Pipeline):
            return self.estimator
        add_estimator_to_pipeline(self.pipeline, self.estimator)
        return self.pipeline
    
    def __exit__(self, type, value, traceback):
        return
    

@dataclass
class Pipeline(Pipeline):
    pipeline: Pipeline
    
    def _fit(self, X, y=None, **fit_params):
        pipeline.fit(X, y, **fit_params)


if __name__ == "__main__":
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    iris = load_iris()
    
    train_df = pd.DataFrame(iris["data"], columns=iris['feature_names'])
    train_df["target"] = iris["target"]
    # https://github.com/pycaret/pycaret/blob/master/pycaret/internal/pipeline.py#L187
    pipeline = Pipeline([])
    pipeline = make_pipeline()
    config = {}
    with transform_pipeline(pipeline, RandomForestClassifier, config) as pipeline:
        pipeline.fit(train_df, train_df["target"])
        pred = pipeline.predict(train_df)
        print(pred)