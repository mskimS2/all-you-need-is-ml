import sys
sys.path.append("../src")
import optuna
import pandas as pd
from models import get_model
from trainer import Trainer

if __name__ == "__main__":
    # test code
    model,args  = get_model("xgboost", "binary_classification")
    train_df = pd.read_csv("dataset/binary_classification.csv")
    train_df.sex = train_df.sex.apply(lambda x: "0" if x == "Male" else "1")
    train_df.sex = train_df.sex.astype(int)
    trainer = Trainer(args)
    trainer.fit(model, train_df, ["age","education.num"], ["sex"], None)