import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict


class BaseModel(ABC):
    @abstractmethod
    def set_up(*args, **kwargs) -> None:
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
    
    @abstractmethod
    def optimize_hyper_params(*args, **kwargs) -> None:
        pass