import numpy as np
from typing import Any, Dict, Union


class RegressorTemplate:
    def __init__(
        self,
        id: str,
        name: str,
        class_def: Any,
        args: Dict[str, Any] = None,
        is_early_stop: bool = True,
        shap: Union[bool, str] = False,
    ) -> None:
        self.shap = shap
        if not args:
            args = {}

        self.is_early_stop = is_early_stop

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:
        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("is_early_stop", self.is_early_stop),
            ("Args", self.args),
        ]

        return dict(d)


class LinearRegressionTemplate(RegressorTemplate):
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
        from sklearn.linear_model import LinearRegression
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="lr",
            name="Linear Regression",
            class_def=LinearRegression,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class LassoRegressionTemplate(RegressorTemplate):
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
        from sklearn.linear_model import Lasso

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="lasso",
            name="Lasso Regression",
            class_def=Lasso,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class RidgeRegressionTemplate(RegressorTemplate):
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
        from sklearn.linear_model import Ridge
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params
        
        super().__init__(
            id="ridge",
            name="Ridge Regression",
            class_def=Ridge,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class ElasticNetTemplate(RegressorTemplate):
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
        from sklearn.linear_model import ElasticNet
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="elastic_net",
            name="Elastic Net",
            class_def=ElasticNet,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class LarsTemplate(RegressorTemplate):
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
        from sklearn.linear_model import Lars

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="least angle regression",
            name="Least Angle Regression",
            class_def=Lars,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        
        
class LassoLarsTemplate(RegressorTemplate):
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
        from sklearn.linear_model import LassoLars

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="lasso least angle regression",
            name="Lasso Least Angle Regression",
            class_def=LassoLars,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        
        
class OrthogonalMatchingPursuitTemplate(RegressorTemplate):
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
        from sklearn.linear_model import OrthogonalMatchingPursuit

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="orthogonal matching pursuit",
            name="Orthogonal Matching Pursuit",
            class_def=OrthogonalMatchingPursuit,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class BayesianRidgeTemplate(RegressorTemplate):
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
        from sklearn.linear_model import BayesianRidge

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="bayesian ridge",
            name="Bayesian Ridge",
            class_def=BayesianRidge,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class AutomaticRelevanceDeterminationTemplate(RegressorTemplate):
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
        from sklearn.linear_model import ARDRegression

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="automatic relevance determination",
            name="Automatic Relevance Determination",
            class_def=ARDRegression,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class PassiveAggressiveRegressorTemplate(RegressorTemplate):
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
        from sklearn.linear_model import PassiveAggressiveRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="passive aggressive regressor",
            name="Passive Aggressive Regressor",
            class_def=PassiveAggressiveRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class RANSACRegressorTemplate(RegressorTemplate):
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
        from sklearn.linear_model import RANSACRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="random sample consensus",
            name="Random Sample Consensus",
            class_def=RANSACRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class TheilSenRegressorTemplate(RegressorTemplate):
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
        from sklearn.linear_model import TheilSenRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="theilsen regressor",
            name="TheilSen Regressor",
            class_def=TheilSenRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class HuberRegressorTemplate(RegressorTemplate):
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
        from sklearn.linear_model import HuberRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="huber",
            name="Huber Regressor",
            class_def=HuberRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class KernelRidgeTemplate(RegressorTemplate):
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
        from sklearn.kernel_ridge import KernelRidge

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="kernel ridge",
            name="Kernel Ridge",
            class_def=KernelRidge,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class SVRModel(RegressorTemplate):
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
        from sklearn.svm import SVR
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="support vector regression",
            name="Support Vector Regression",
            class_def=SVR,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class KNeighborsRegressorTemplate(RegressorTemplate):
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
        from sklearn.neighbors import KNeighborsRegressor
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="k neighbors regressor",
            name="K Neighbors Regressor",
            class_def=KNeighborsRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class DecisionTreeRegressorTemplate(RegressorTemplate):
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
        from sklearn.tree import DecisionTreeRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="decision tree regressor",
            name="Decision Tree Regressor",
            class_def=DecisionTreeRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class RandomForestRegressorTemplate(RegressorTemplate):
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
        from sklearn.ensemble import RandomForestRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="random forest regressor",
            name="Random Forest Regressor",
            class_def=RandomForestRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        
        
class ExtraTreesRegressorTemplate(RegressorTemplate):
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
        from sklearn.ensemble import ExtraTreesRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="extra trees regressor",
            name="Extra Trees Regressor",
            class_def=ExtraTreesRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class AdaBoostRegressorTemplate(RegressorTemplate):
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
        from sklearn.ensemble import AdaBoostRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="adaboost",
            name="AdaBoost Regressor",
            class_def=AdaBoostRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class GradientBoostingRegressorTemplate(RegressorTemplate):
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
        from sklearn.ensemble import GradientBoostingRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="gradient boosting regressor",
            name="Gradient Boosting Regressor",
            class_def=GradientBoostingRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class MLPRegressorTemplate(RegressorTemplate):
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
        from sklearn.neural_network import MLPRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="mlp",
            name="MLP Regressor",
            class_def=MLPRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

class XGBRegressorTemplate(RegressorTemplate):
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
        from xgboost import XGBRegressor
        
        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="xgboost",
            name="XGBoost Regressor",
            class_def=XGBRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )


class LGBMRegressorTemplate(RegressorTemplate):
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
        from lightgbm import LGBMRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        
        
class CatBoostRegressorTemplate(RegressorTemplate):
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
        from catboost import CatBoostRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="catboost",
            name="CatBoost Regressor",
            class_def=CatBoostRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )

 
class DummyRegressorTemplate(RegressorTemplate):
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
        from sklearn.dummy import DummyRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="dummy",
            name="Dummy Regressor",
            class_def=DummyRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

# FIXME: This is not working
class BaggingRegressorTemplate(RegressorTemplate):
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
        from sklearn.ensemble import BaggingRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="Bagging",
            name="Bagging Regressor",
            class_def=BaggingRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

# FIXME: This is not working
class StackingRegressorTemplate(RegressorTemplate):
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
        from sklearn.ensemble import StackingRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="Stacking",
            name="Stacking Regressor",
            class_def=StackingRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )
        

# FIXME: This is not working
class VotingRegressorTemplate(RegressorTemplate):
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
        from sklearn.ensemble import VotingRegressor

        self.logger = logger
        self.use_gpu = use_gpu
        self.tune_params = tune_params

        super().__init__(
            id="voting regressor",
            name="Voting Regressor",
            class_def=VotingRegressor,
            args=args,
            shap=shap,
            is_ealry_stop=is_ealry_stop,
        )