import numpy as np
from typing import Any, Dict, Union


class ClassifierTemplate:
    def __init__(
        self,
        id: str,
        name: str,
        class_def: Any,
        is_early_stop: bool = True,
        args: Dict[str, Any] = None,
        shap: Union[bool, str] = False,
    ) -> None:
        self.id = id
        self.name = name
        self.shap = shap
        self.model = class_def
        if args:
           self.model(**args)

        self.is_early_stop = is_early_stop

    def get_dict(self) -> Dict[str, Any]:
        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("Class", self.class_def),
            ("is_early_stop", self.is_early_stop),
            ("Args", self.args),
        ]

        return dict(d)


class LogisticRegressionClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment, 
        logger: Any, 
        use_gpu: bool = False,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="lr",
            name="Logistic Regression",
            class_def=LogisticRegression,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class KNeighborsClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.neighbors import KNeighborsClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="knn",
            name="KNeighborsClassifier",
            class_def=KNeighborsClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class GaussianNBClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.naive_bayes import GaussianNB

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="nb",
            name="Naive Bayes(GaussianNB)",
            class_def=GaussianNB,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class DecisionTreeClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.tree import DecisionTreeClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="dt",
            name="Decision Tree",
            class_def=DecisionTreeClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class SGDClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.linear_model import SGDClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="svm",
            name="SVM - Linear Kernel",
            class_def=SGDClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class SVCClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.svm import SVC
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="rbfsvm",
            name="SVM - Radial Kernel",
            class_def=SVC,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class GaussianProcessClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.gaussian_process import GaussianProcessClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="gpc",
            name="Gaussian Process Classifier",
            class_def=GaussianProcessClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class MLPClassifierTemplate(ClassifierTemplate):
    def __init__(
        self, 
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.neural_network import MLPClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="mlp",
            name="MLP Classifier",
            class_def=MLPClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class RidgeClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.linear_model import RidgeClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="ridge",
            name="Ridge Classifier",
            class_def=RidgeClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class RandomForestClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="rf",
            name="Random Forest Classifier",
            class_def=RandomForestClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )

class QuadraticDiscriminantAnalysisTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="qda",
            name="Quadratic Discriminant Analysis",
            class_def=QuadraticDiscriminantAnalysis,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )

class AdaBoostClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.ensemble import AdaBoostClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="ada",
            name="Ada Boost Classifier",
            class_def=AdaBoostClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )        

class GradientBoostingClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="gbc",
            name="Gradient Boosting Classifier",
            class_def=GradientBoostingClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )

class LinearDiscriminantAnalysisTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params
        
        super().__init__(
            id="lda",
            name="Linear Discriminant Analysis",
            class_def=LinearDiscriminantAnalysis,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        
class ExtraTreesClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.ensemble import ExtraTreesClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="et",
            name="Extra Trees Classifier",
            class_def=ExtraTreesClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )

class XGBClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from xgboost import XGBClassifier
         
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="xgboost",
            name="Extreme Gradient Boosting",
            class_def=XGBClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class LGBMClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from lightgbm import LGBMClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class CatBoostClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from catboost import CatBoostClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="catboost",
            name="CatBoost Classifier",
            class_def=CatBoostClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class DummyClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.dummy import DummyClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="dummy",
            name="Dummy Classifier",
            class_def=DummyClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class BaggingClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.ensemble import BaggingClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="Bagging",
            name="Bagging Classifier",
            class_def=BaggingClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class StackingClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.ensemble import StackingClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="Stacking",
            name="Stacking Classifier",
            class_def=StackingClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        
        
# FIXME: This is not working
class VotingClassifierTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.ensemble import VotingClassifier

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="Voting",
            name="Voting Classifier",
            class_def=VotingClassifier,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class CalibratedClassifierCVTemplate(ClassifierTemplate):
    def __init__(
        self,
        experiment,
        logger: Any,
        args: Dict[str, Any] = {},
        tune_params: Dict[str, Any] = {},
        use_gpu: bool = False,
        shap: Union[bool, str] = False,
        is_ealry_stop: bool = True,
    ) -> None:
        from sklearn.calibration import CalibratedClassifierCV

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="CalibratedCV",
            name="Calibrated Classifier CV",
            class_def=CalibratedClassifierCV,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )