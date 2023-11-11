from const import Const
from typing import List

class Task:
    type = {
        Const.CLASSIFICATION: 0,
        Const.REGRESSION: 1,
    }
        
    @staticmethod
    def from_str(t: str) -> int:
        if Task.type.get(t) is not None:
            return Task.type[t]
        
        raise ValueError(f"Invalid task type: {t}")

    @staticmethod
    def list_str() -> List[str]:
        return list(Task.type.keys())
    

class Problem:
    type = {
        Const.BINARY_CLASSIFICATION: 1,
        Const.MULTI_CLASS_CLASSIFICATION: 2,
        Const.MULTI_LABEL_CLASSIFICATION: 3,
        Const.SINGLE_COLUMN_REGRESSION: 4,
        Const.MULTI_COLUMN_REGRESSION: 5,
    }
        
    @staticmethod
    def from_str(t: str) -> int:
        if Problem.type.get(t) is not None:
            return Problem.type[t]
        
        raise ValueError(f"Invalid problem type: {t}")

    @staticmethod
    def list_str() -> List[str]:
        return list(Problem.type.keys())