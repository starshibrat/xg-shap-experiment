from typing import Callable, Tuple
import numpy as np
from tree.single_hist_tree_boost import SingleHistTreeBoost
from objective.logistic_classification import LogisticClassification
from sklearn.metrics import log_loss
from common import common

class MiniXGBoostClassifier:
    def __init__(self,
        objective: str | Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = "binary:logistic",
        max_bin: np.uint32 = 256,
        max_depth: np.uint16 = 6,
        tree_method: str = "hist",
        learning_rate: np.float32 = 0.5,
        ALPHA: np.float32 = 0.,
        LAMBDA: np.float32 = 0.,
        GAMMA: np.float32 = 0.,
        n_estimators: int = 100,
        custom_loss: Callable[[np.ndarray, np.ndarray], np.float32] = None,
        verbose: bool = False,
        eval_param: dict = dict(),
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.ALPHA = ALPHA
        self.LAMBDA = LAMBDA
        self.GAMMA = GAMMA
        self.n_estimators = n_estimators
        self.trees = []
        self.tree_method = tree_method
        self.max_bin = max_bin
        self.objective = objective
        self.custom_loss = custom_loss
        self.verbose = verbose
        self.eval_param = eval_param

    def _get_grad_and_hess(self, label: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.objective, str):
            match self.objective:
                case "binary:logistic":
                    obj = LogisticClassification(label, y_pred)

                    return obj.first_order_gradient(), obj.second_order_gradient()
        else:
            return self.objective(label, y_pred)
    
    def _get_loss_metrics(self, label: np.ndarray, y_pred: np.ndarray):
        if self.custom_loss:
            # p  = common.sigmoid(label)
            return self.custom_loss(label, y_pred)

        match self.objective:
            case "binary:logistic":
                return log_loss(label, y_pred)
            case _:
                return log_loss(label, y_pred)

    def fit(self, X: np.ndarray, y: np.ndarray):
        strong_pred = np.zeros(len(y))
        for i in range(self.n_estimators):
            p = common.sigmoid(strong_pred)

            grad, hess = self._get_grad_and_hess(y, p)

            match self.tree_method:
                case "hist":
                    tree = SingleHistTreeBoost(
                        max_depth=self.max_depth,
                        grad=grad,
                        hess=hess,
                        max_bin=self.max_bin,
                    )
                case _:
                    tree = SingleHistTreeBoost(
                        max_depth=self.max_depth,
                        grad=grad,
                        hess=hess,
                        max_bin=self.max_bin,
                    )

            tree.fit(X)
            new_pred = tree.predict(X)
            strong_pred += self.learning_rate * new_pred
            self.trees.append(tree)
            
            loss = self._get_loss_metrics(y, p)
            if self.verbose:
                print(f"Iteration: {i+1} | loss = {loss:.3f}")

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        pred = np.zeros(len(X))
        for t in self.trees:
            pred += t.predict(X) * self.learning_rate
        return pred
    

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return common.sigmoid(self.predict_raw(X))
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def eval_metrics(param: dict) -> dict:
        res = dict()

