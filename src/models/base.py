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


def get_all_containers(
    container_globals: dict,
    experiment: Any,
    type_var: type,
    raise_errors: bool = True,
) -> Dict[str, BaseContainer]:
    # https://stackoverflow.com/a/1401900/8925915
    model_container_classes = [
        obj
        for _, obj in container_globals.items()
        if inspect.isclass(obj)
        # Get all parent class types excluding the object class type
        # If this is not excluded, then containers like TimeSeriesContainer
        # also shows up in model_container_classes
        and type_var in tuple(x for x in inspect.getmro(obj) if x != obj)
    ]

    model_containers = []

    for obj in model_container_classes:
        if raise_errors:
            if hasattr(obj, "active") and not obj.active:
                continue
            instance = obj(experiment)
            if instance.active:
                model_containers.append(instance)
        else:
            try:
                if hasattr(obj, "active") and not obj.active:
                    continue
                instance = obj(experiment)
                if instance.active:
                    model_containers.append(instance)
            except Exception:
                pass

    return {container.id: container for container in model_containers}