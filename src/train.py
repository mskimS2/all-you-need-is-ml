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
    
    # test code
    xgb, xgb_args = get_model("xgboost", "binary_classification")
    lgbm, lgbm_args = get_model("lightgbm", "binary_classification")
    cat, cat_args = get_model("catboost", "binary_classification")
    dt, dt_args = get_model("decision_tree", "binary_classification")
    rf, rf_args = get_model("random_forest", "binary_classification")
    et, et_args = get_model("extra_tree", "binary_classification")
    sgd, sgd_args = get_model("sgd_classifier", "binary_classification")
    
    train_df = pd.read_csv("dataset/binary_classification.csv")
    train_df.sex = train_df.sex.apply(lambda x: "0" if x == "Male" else "1").astype(int)
    test_df = None
    
    encoder = Encoder(encoder=LabelEncoder())
    
    # trainer = Trainer(xgb, xgb_args, scaler=None, encoder=encoder)
    # trainer.fit(train_df, test_df, ["age","education.num"], ["sex"])
    # print(trainer.feature_importacne())
    
    # trainer = Trainer(lgbm, lgbm_args, scaler=None, encoder=encoder)
    # trainer.fit(train_df, test_df, ["age","education.num"], ["sex"])
    # print(trainer.feature_importacne())
    
    # trainer = Trainer(cat, cat_args, scaler=None, encoder=encoder)
    # trainer.fit(train_df, test_df, ["age","education.num"], ["sex"])
    # print(trainer.feature_importacne())
    
    # trainer = Trainer(dt, dt_args, scaler=None, encoder=encoder)
    # trainer.fit(train_df, test_df, ["age","education.num"], ["sex"])
    # print(trainer.feature_importacne())
    
    # trainer = Trainer(rf, rf_args, scaler=None, encoder=encoder)
    # trainer.fit(train_df, test_df, ["age","education.num"], ["sex"])
    # print(trainer.feature_importacne())
    
    trainer = Trainer(et, et_args, scaler=None, encoder=encoder)
    trainer.fit(train_df, test_df, ["age","education.num"], ["sex"])
    print(trainer.feature_importacne())
    
    trainer = Trainer(sgd, sgd_args, scaler=None, encoder=encoder)
    trainer.fit(train_df, test_df, ["age","education.num"], ["sex"])
    print(trainer.feature_importacne())
