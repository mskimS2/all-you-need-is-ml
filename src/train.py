import sys
sys.path.append("../src")
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from trainer import Trainer
from preprocessing import Encoder
from models.model import get_model


if __name__ == "__main__":
    # test code
    model, args  = get_model("xgboost", "binary_classification")
    train_df = pd.read_csv("dataset/binary_classification.csv")
    train_df.sex = train_df.sex.apply(lambda x: "0" if x == "Male" else "1")
    train_df.sex = train_df.sex.astype(int)
    
    encoder = Encoder(encoder=LabelEncoder())
    trainer = Trainer(model, args, scaler=None, encoder=encoder)
    trainer.fit(train_df, ["age","education.num"], ["sex"])
    print(trainer.feature_importacne())