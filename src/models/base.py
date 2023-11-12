import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def setup(*args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def fit(*args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def predict(*args, **kwargs) -> np.ndarray:
        pass
    
    @abstractmethod
    def feature_importances(*args, **kwargs) -> pd.DataFrame:
        pass