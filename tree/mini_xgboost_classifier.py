from typing import Callable, Tuple
import numpy as np
from tree.vectorized_single_hist_tree_boost import VectorizedSingleHistTreeBoost
from objective.logistic_classification import LogisticClassification
from sklearn.metrics import log_loss
from common import common
import concurrent.futures
import os
from tree.single_tree import SingleTree

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
        categorical_features: list[int]= [],
        scale_pos_weight=1.0,
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
        self.categorical_features = categorical_features
        self.scale_pos_weight = scale_pos_weight

    def _quantize_features(self, X: np.ndarray):
        nan_mask = np.isnan(X)
        nan_indices = np.argwhere(nan_mask)[:, 1]
        column_min_val = np.nanmin(X, axis=0)

        Xc = X.copy()
        Xc[nan_mask] = column_min_val[nan_indices]

        self.bin_categorical_features = np.bincount(self.categorical_features, minlength=X.shape[1]).astype(bool)
        percentiles = np.linspace(0, 100, self.max_bin + 1)
        bin_edges = np.percentile(Xc, percentiles, axis=0)

        X_binned = np.zeros_like(Xc, dtype=np.int32)
        for j in range(X.shape[1]):
            if self.bin_categorical_features[j]:
                X_binned[:, j] = Xc[:, j]
                continue

            bins = np.digitize(Xc[:, j], bin_edges[:, j], right=True) - 1
            X_binned[:, j] = np.clip(bins, 0, self.max_bin - 1)
        
        X_binned[nan_mask] = self.max_bin

        if self.verbose:
            print("X_binned=", X_binned)
            print("bin_edges=", bin_edges)
        
        return X_binned, bin_edges

    def _get_grad_and_hess(self, label: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(self.objective, str):
            match self.objective:
                case "binary:logistic":
                    obj = LogisticClassification(label, y_pred)

                    g = obj.first_order_gradient()
                    h = obj.second_order_gradient()

                    if self.scale_pos_weight != 1.0:
                        positive_mask = (label == 1)
                        g[positive_mask] *= self.scale_pos_weight
                        h[positive_mask] *= self.scale_pos_weight

                    return g, h
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

    def fit(self, X: np.ndarray, y: np.ndarray, show_metrics=False):
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
                        min_child_weight=self.min_child_weight,
                        categorical_features=self.categorical_features,
                        verbose=self.verbose,
                    )
                case "single":
                    tree = SingleTree(
                        max_depth=self.max_depth,
                        grad=grad,
                        hess=hess,
                        max_bin=self.max_bin,
                        LAMBDA=self.LAMBDA,
                        GAMMA=self.GAMMA,
                        min_child_weight=self.min_child_weight,
                        categorical_features=self.categorical_features,
                        verbose=self.verbose,
                    )

            tree.fit(X_binned, bin_edges)
            new_pred = tree.predict_train(len(y))
            strong_pred += self.learning_rate * new_pred
            tree.clean_up_tree()
            self.trees.append(tree)
            
            loss = self._get_loss_metrics(y, p)
            if show_metrics:
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
    
    def parallel_explain(self, X: np.ndarray, max_workers: int = 0, batch_size: int = 1000) -> Tuple[np.ndarray, np.float32]:
        n_samples = X.shape[0]
        expected_value = 0.0
        total_shap_value = np.zeros_like(X, dtype=np.float32)

        if max_workers == 0:
            max_workers = max(1, os.cpu_count() - 1)
        
        for i in range(0, n_samples, batch_size):
            X_batch = X[i : i + batch_size]
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = {
                    executor.submit(t.calculate_shap, X_batch): t for t in self.trees
                }

                for future in concurrent.futures.as_completed(futures):
                    shap_val, exp_val = future.result()
                    total_shap_value[i : i + batch_size] += self.learning_rate * shap_val
                    
                    if i == 0:
                        expected_value += self.learning_rate * exp_val

            print(f"batch {i}~{min(i + batch_size, n_samples)} done.")
        
        return total_shap_value, expected_value
        
