import sys
sys.path.append('../src')
import pandas as pd
import numpy as np
import optuna
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from functools import partial
from typing import List, Any

def optimize_hyper_parameters(
    self, 
    df: pd.DataFrame, 
    features: List[str] = None,
    targets: List[str] = None,
    n_trials: int = 15,
    direction: str = "minimize",
) -> dict[str, Any]:             
    study = optuna.create_study(direction=direction)
    study.optimize(
        partial(
            optimize_hyperparams, 
            df=df, 
            features=features, 
            targets=targets,
        ), 
        n_trials=n_trials,
    )
    
    return study.best_params
    

def optimize_hyperparams(trial, x, y):
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trial.suggest_int("max_depth", 3, 15)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)
    
    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        criterion=criterion,
    )
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuaraies = []
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = X[train_idx]
        ytrain = y[train_idx]
        xtest = X[test_idx]
        ytest = y[test_idx]
        
        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuaraies.append(fold_acc)
    
    return -1.0 * np.mean(accuaraies)

if __name__ == "__main__":
    df = pd.read_csv("dataset/mobile_train.csv")
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values
    
    """grid search"""
    # classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    # param_grid = {
    #     "n_estimators": [100, 200, 300, 400],
    #     "max_depth": [1, 3, 5, 7],
    #     "criterion": ["gini", "entropy"],
    # }
    
    # model = model_selection.GridSearchCV(
    #     estimator=classifier,
    #     param_grid=param_grid,
    #     scoring="accuracy",
    #     verbose=10,
    #     n_jobs=1,
    #     cv=5,
    # )
    
    # model.fit(X, y)
    
    # print(model.best_score_)
    # print(model.best_estimator_.get_params())
    
    # classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    # param_grid = {
    #     "n_estimators": np.arange(100, 1500, 100),
    #     "max_depth": np.arange(1, 20),
    #     "criterion": ["gini", "entropy"],
    # }
    
    """random search"""
    # model = model_selection.RandomizedSearchCV(
    #     estimator=classifier,
    #     param_distributions=param_grid,
    #     n_iter=10,
    #     scoring="accuracy",
    #     verbose=10,
    #     n_jobs=1,
    #     cv=5,
    # )
    
    # model.fit(X, y)
    
    # print(model.best_score_)
    # print(model.best_estimator_.get_params())
    
    """pipeline search"""
    # scl = preprocessing.StandardScaler()
    # pca = decomposition.PCA()
    # rf = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # classifier = pipeline.Pipeline(
    #     [
    #         ("scaling", scl),
    #         ("pca", pca),
    #         ("rf", rf),
    #     ]
    # )
    
    # param_grid = {
    #     "pca__n_components": np.arange(5, 10),
    #     "rf__n_estimators": np.arange(100, 1500, 100),
    #     "rf__max_depth": np.arange(1, 20),
    #     "rf__criterion": ["gini", "entropy"],
    # }
    
    # model = model_selection.RandomizedSearchCV(
    #     estimator=classifier,
    #     param_distributions=param_grid,
    #     n_iter=10,
    #     scoring="accuracy",
    #     verbose=10,
    #     n_jobs=1,
    #     cv=5,
    # )
    
    # model.fit(X, y)
    
    # print(model.best_score_)
    # print(model.best_estimator_.get_params())
    
    """optimize function"""
    # param_space = [
    #     space.Integer(3, 15, name="max_depth"),
    #     space.Integer(100, 600, name="n_estimators"),
    #     space.Categorical(["gini", "entropy"], name="criterion"),
    #     space.Real(0.01, 1, prior="uniform", name="max_features"),
    # ]
    # params_names = [
    #     "max_depth",
    #     "n_estimators",
    #     "criterion",
    #     "max_features"
    # ]
    # optimization_function = partial(
    #     optimize,
    #     param_names=params_names,
    #     x=X,
    #     y=y
    # )
    
    # result = gp_minimize(
    #     optimization_function,
    #     dimensions=param_space,
    #     n_calls=15,
    #     n_random_starts=10,
    #     verbose=10,
    # )
    
    # print(dict(zip(params_names,result.x)))
    
    """optuna"""
    optimization_function = partial(optuna_optimize, x=X, y=y)
    study = optuna.create_study(direction="minimize")
    trial = study.optimize(optimization_function, n_trials=15)
    
    # study.best_params
    