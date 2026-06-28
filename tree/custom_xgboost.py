import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

class CustomXGBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, 
        max_depth=3, 
        learning_rate=0.1, 
        n_estimators=100,
        enable_categorical=True,
        random_state=1,
        reg_lambda=1e-9,
        max_cat_to_onehot=1,
        min_child_weight=0,
        objective='binary:logistic',
        base_score=0.5,
        verbose=0,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.enable_categorical = enable_categorical
        self.random_state=random_state
        self.reg_lambda=reg_lambda
        self.max_cat_to_onehot=max_cat_to_onehot
        self.min_child_weight=min_child_weight
        self.objective=objective
        self.base_score=base_score
        self.verbose=verbose
        

    def fit(self, X, y, **fit_params):
        # Dynamically calculate the weight based ONLY on the current fold's training y
        self.classes_ = np.unique(y)
        _dynamic_weight = np.sum(y == 0) / np.sum(y == 1)

        if self.verbose > 0:
            print("==="*10)
            print("Current params")
            print(self.get_params())
            print("Current X")
            print(X)
            print("Current y")
            print(y)
            print("==="*10)
        
        self.model = XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            scale_pos_weight=_dynamic_weight,
            enable_categorical=self.enable_categorical,
            random_state=self.random_state,
            reg_lambda=self.reg_lambda,
            max_cat_to_onehot=self.max_cat_to_onehot,
            min_child_weight=self.min_child_weight,
            objective=self.objective,
            base_score=self.base_score,
        )

        self.model.fit(X, y, **fit_params)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)