from typing import List


class Task:
    type = {
        "classification": 0,
        "regression": 1,
    }
        
    @staticmethod
    def from_str(t: str) -> int:
        if Task.type.get(t) is not None:
            return Task.type[t]
        
        raise ValueError(f"Invalid task type: {t}")

    @staticmethod
    def list_str() -> List[str]:
        return Task.type.keys()
    

class Problem:
    type = {
        "binary_classification": 1,
        "multi_class_classification": 2,
        "multi_label_classification": 3,
        "single_column_regression": 4,
        "multi_column_regression": 5,
    }
        
    @staticmethod
    def from_str(t: str) -> int:
        if Problem.type.get(t) is not None:
            return Problem.type[t]
        
        raise ValueError(f"Invalid problem type: {t}")

    @staticmethod
    def list_str() -> List[str]:
        return Problem.type.keys()

if __name__ == "__main__":
    print(Task.from_str("classification"))
    print(Task.from_str("regression"))
    print(Task.list_str())
    
    print(Problem.from_str("binary_classification"))
    print(Problem.from_str("multi_class_classification"))
    print(Problem.from_str("multi_label_classification"))
    print(Problem.from_str("single_column_regression"))
    print(Problem.from_str("multi_column_regression"))
    
    print(Problem.list_str())