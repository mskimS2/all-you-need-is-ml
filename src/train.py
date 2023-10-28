import os
import sys
sys.path.append("/C:/workspace/awesome-tablet-data/src")
print(sys.path)
import os
import optuna
import pandas as pd
from typing import List
from logger import logger
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