from typing import Callable, Tuple
import numpy as np
from objective.logistic_classification import LogisticClassification
from sklearn.metrics import log_loss
from common import common
import concurrent.futures
import os
from tree.single_tree import SingleTree
import xgboost as xgb

class XGClassifier:
    def __init__(self,
        objective: str | Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = "binary:logistic",
        max_bin: np.uint32 = 256,
        max_depth: np.uint16 = 6,
        learning_rate: np.float32 = 0.5,
        ALPHA: np.float32 = 0.,
        reg_lambda: np.float32 = 0.,
        min_split_loss: np.float32 = 0.,
        n_estimators: int = 100,
        custom_loss: Callable[[np.ndarray, np.ndarray], np.float32] = None,
        min_child_weight: float = 1.0,
        verbose: bool = False,
        vectorized: bool = False,
        categorical_features: list[int]= [],
        scale_pos_weight=1.0,
        base_score: str | float | int = None,
        
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.ALPHA = ALPHA
        self.reg_lambda = reg_lambda
        self.min_split_loss = min_split_loss
        self.n_estimators = n_estimators
        self.trees = []
        self.max_bin = max_bin
        self.objective = objective
        self.custom_loss = custom_loss
        self.min_child_weight = min_child_weight
        self.verbose = verbose
        self.vectorized = vectorized
        self.categorical_features = categorical_features
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score

    @classmethod
    def load_from_xgbclassifier(cls, model: xgb.XGBClassifier, columns: list[str]) -> XGClassifier:
        booster = model.get_booster()
        trees = []

        for i in range(booster.num_boosted_rounds()):
            st = SingleTree.load_from_booster(booster[i], columns)
            trees.append(st)
        
        m = XGClassifier()
        m.trees = trees

        return m

    def _quantize_features(self, X: np.ndarray):
        nan_mask = np.isnan(X)

        column_max_val = np.nanmax(X, axis=0).astype(np.uint8) + 1
        X_copy = X.copy()
        X_copy[nan_mask] = np.take(column_max_val, np.where(nan_mask)[1])

        return X_copy.astype(np.uint8), column_max_val

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

                    if self.verbose:
                        print(f"\ny_pred\n{y_pred}\n")
                        print(f"\np\n{obj.p}\n")
                        print(f"\nGradient\n{g}\n")
                        print(f"\nHessian\n{h}\n")

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
            
    def init_base_score(self, y: np.ndarray) -> np.ndaray:

        N_pos = np.sum(y == 1)
        N_neg = np.sum(y == 0)

        sum_w_pos = N_pos * self.scale_pos_weight
        sum_w_neg = N_neg * 1.0

        p_init = sum_w_pos / (sum_w_pos + sum_w_neg)

        if p_init == 0.0:
            initial_score = -np.inf
        elif p_init == 1.0:
            initial_score = np.inf
        else:
            initial_score = common.logit(p_init)

        return np.full(shape=y.shape, fill_value=initial_score, dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, show_metrics=False):
        if (isinstance(self.base_score, str) and self.base_score == "intercept") or not self.base_score:
            strong_pred = self.init_base_score(y)
        elif isinstance(self.base_score, int) or isinstance(self.base_score, float):
            strong_pred = np.full_like(y, common.logit(self.base_score), dtype=np.float32)
        
        X_binned, column_max_val = self._quantize_features(X)

        for i in range(self.n_estimators):
            grad, hess = self._get_grad_and_hess(y, strong_pred)

            tree = SingleTree(
                max_depth=self.max_depth,
                grad=grad,
                hess=hess,
                reg_lambda=self.reg_lambda,
                min_split_loss=self.min_split_loss,
                min_child_weight=self.min_child_weight,
                categorical_features=self.categorical_features,
                verbose=self.verbose,
                column_max_val=column_max_val,
                learning_rate=self.learning_rate,
            )
            
            if show_metrics:
                p = common.sigmoid(strong_pred)
                loss = self._get_loss_metrics(y, p)
                print(f"Iteration: {i+1} | loss = {loss:.3f}")

            tree.fit(X_binned)
            new_pred = tree.predict_train(len(y))
            strong_pred += new_pred
            tree.clean_up_tree()
            self.trees.append(tree)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        pred = np.zeros(len(X))
        if self.verbose:
            print("\n===PREDICT INIT===")
            print(f"\npred={pred}\n")
        for i in range(len(self.trees)):
            t = self.trees[i]
            pred += t.predict(X)
            if self.verbose:
                    print(f"\n{i + 1}th tree")
                    print(f"\nres={pred}\n")
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        res = common.sigmoid(self.predict_raw(X))
        if self.verbose:
            print(f"\nsigmoid'd pred raw={res}")
        return res
    
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
                    total_shap_value[i : i + batch_size] += shap_val
                    
                    if i == 0:
                        expected_value += exp_val

            print(f"batch {i}~{min(i + batch_size, n_samples)} done.")
        
        return total_shap_value, expected_value