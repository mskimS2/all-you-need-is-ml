import optuna


class CategoricalDistribution(optuna.distributions.BaseDistribution):
    """Categorical distribution for Optuna"""

    def __init__(self, values):
        self.values = values
    
    def get_sample(self):
        return optuna.distributions.CategoricalDistribution(self.values)
    
    def __repr__(self):
        return f"Optuna Distribution values={self.values}"