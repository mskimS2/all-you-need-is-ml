import sys
sys.path.append("../src")
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from trainer import Trainer
from preprocessor import Encoder
from models.factory import get_model
from utils import set_randomness


if __name__ == "__main__":
    set_randomness(2023)
    
    # binary classification test code
    # encoder = Encoder(encoder=LabelEncoder())
    # problem = "binary_classification"
    # for model_name  in [
    #     "xgboost",
    #     "lightgbm",
    #     "catboost",
    #     "decision_tree",
    #     "random_forest",
    #     "extra_tree",
    #     "support_vector_machine",
    #     "knn",
    # ]:
    #     train_df = pd.read_csv("dataset/binary_classification.csv")
    #     train_df.sex = train_df.sex.apply(lambda x: "0" if x == "Male" else "1").astype(int)
    #     test_df = None
    
    #     model, config = get_model(model_name, problem)
    #     trainer = Trainer(model, config, scaler=None, encoder=encoder)
    #     trainer.fit(train_df=train_df, test_df=test_df, features=["age","education.num"], targets=["sex"])
    #     print(trainer.feature_importance())  
    
    from sklearn.datasets import load_iris
    iris = load_iris()
    
    encoder = Encoder(encoder=LabelEncoder())
    
    problem = "single_column_regression"
    for model_name  in [
        "xgboost",
        "lightgbm",
        "catboost",
        "decision_tree",
        "random_forest",
        "extra_tree",
        "logistic_regression",
        "lasso",
        "linear_regression",
        "support_vector_machine",
        "knn",
    ]:
        train_df = pd.DataFrame(iris["data"], columns=iris['feature_names'])
        train_df["target"] = iris["target"]
        test_df = None
    
        model, config = get_model(model_name, problem)
        trainer = Trainer(model, config, scaler=None, encoder=encoder)
        trainer.fit(train_df=train_df, test_df=test_df, features=iris['feature_names'], targets=["target"])
        print(trainer.feature_importance())