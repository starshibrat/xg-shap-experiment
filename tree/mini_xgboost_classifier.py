from typing import Callable, Tuple
import numpy as np
from tree.single_hist_tree_boost import SingleHistTreeBoost
from tree.vectorized_single_hist_tree_boost import VectorizedSingleHistTreeBoost
from objective.logistic_classification import LogisticClassification
from sklearn.metrics import log_loss
from common import common
import concurrent.futures
from multiprocessing import shared_memory
import os
import threading

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
        min_child_weight: float = 1.0,
        verbose: bool = False,
        vectorized: bool = False,
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
        self.min_child_weight = min_child_weight
        self.verbose = verbose
        self.vectorized = vectorized

    def _quantize_features(self, X):

        percentiles = np.linspace(0, 100, self.max_bin + 1)
        bin_edges = np.percentile(X, percentiles, axis=0)

        X_binned = np.zeros_like(X, dtype=np.int32)
        for j in range(X.shape[1]):
            bins = np.digitize(X[:, j], bin_edges[:, j], right=True) - 1
            X_binned[:, j] = np.clip(bins, 0, self.max_bin - 1)
        
        return X_binned, bin_edges

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
            return self.custom_loss(label, y_pred)

        match self.objective:
            case "binary:logistic":
                return log_loss(label, y_pred)
            case _:
                return log_loss(label, y_pred)

    def fit(self, X: np.ndarray, y: np.ndarray):
        strong_pred = np.zeros(len(y))
        X_binned, bin_edges = self._quantize_features(X)

        for i in range(self.n_estimators):
            p = common.sigmoid(strong_pred)

            grad, hess = self._get_grad_and_hess(y, strong_pred)

            match self.tree_method:
                case "hist":
                    tree = VectorizedSingleHistTreeBoost(
                        max_depth=self.max_depth,
                        grad=grad,
                        hess=hess,
                        max_bin=self.max_bin,
                        LAMBDA=self.LAMBDA,
                        GAMMA=self.GAMMA,
                        min_child_weight=self.min_child_weight
                    )
                case _:
                    tree = VectorizedSingleHistTreeBoost(
                        max_depth=self.max_depth,
                        grad=grad,
                        hess=hess,
                        max_bin=self.max_bin,
                        LAMBDA=self.LAMBDA,
                        GAMMA=self.GAMMA,
                        min_child_weight=self.min_child_weight
                    )

            tree.fit(X_binned, bin_edges)
            new_pred = tree.predict_train(len(y))
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

    def explain(self, X: np.ndarray) -> Tuple[np.ndarray, np.float32]:
        expected_value = 0
        total_shap_value = np.zeros_like(X)

        for t in self.trees:
            shap_val, exp_val = t.calculate_shap(X)
            total_shap_value += self.learning_rate * shap_val
            expected_value += self.learning_rate * exp_val
        
        return total_shap_value, expected_value
    
    def parallel_explain(self, X: np.ndarray, max_workers: int = 0) -> Tuple[np.ndarray, np.float32]:
        expected_value = 0
        total_shap_value = np.zeros_like(X)

        if max_workers == 0:
            max_workers = os.cpu_count() - 1
        
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = {
                executor.submit(t.calculate_shap, X): t for t in self.trees
            }

            for future in concurrent.futures.as_completed(futures):
                shap_val, exp_val = future.result()
                total_shap_value += self.learning_rate * shap_val
                expected_value += self.learning_rate * exp_val
        
        return total_shap_value, expected_value
        
