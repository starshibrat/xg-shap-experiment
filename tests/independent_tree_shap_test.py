from mini_shap.independent_tree_shap import IndependentTreeExplainer
import pandas as pd
import unittest
from tree.xg_classifier import XGClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime

class TestTreeSHAP(unittest.TestCase):

    def load_data_and_scale_pos_weight(self, size: int):
        current_dir = Path(__file__).parent
        file_path = current_dir / "../datasets/BRFSS/out_v2.parquet"
        df = pd.read_parquet(file_path.resolve())
        limit = size

        y = df.iloc[:limit, 0]
        X = df.iloc[:limit, 1:].astype("category")

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

        classes = y_train.value_counts()
        scale_pos_weight = classes[0]/classes[1]

        X_train_np, X_test_np, y_train_np, y_test_np = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32)

        return X_train_np, X_test_np, y_train_np, y_test_np, scale_pos_weight
    
    def load_parameter(self, max_depth=1, learning_rate=0.1, n_estimators=3, gamma=0, reg_lambda=1, base_score=0.5):
        max_depth = max_depth
        learning_rate = learning_rate
        n_estimators = n_estimators
        gamma = gamma
        lb = reg_lambda
        base_score = base_score

        return (max_depth, learning_rate, n_estimators, gamma, lb, base_score)

    @unittest.skip
    def test_local_accuracy_single(self):
        X_train_np, X_test_np, y_train_np, y_test_np, scale_pos_weight = self.load_data_and_scale_pos_weight(50000)

        max_depth, learning_rate, n_estimators, gamma, lb, base_score = self.load_parameter(max_depth=2)

        dt2 = XGClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=lb,
            min_child_weight=gamma,
            categorical_features=[i for i in range(X_train_np.shape[1])],
            base_score=base_score,
            verbose=False
        )

        dt2.fit(X_train_np, y_train_np)

        foreground_index = 400
        foreground_sample = X_test_np[foreground_index]

        base_val = dt2.predict_raw(X_train_np).mean()
        shap_sum = base_val

        start_time = datetime.now()

        for i in range(n_estimators):
            tree_exp = IndependentTreeExplainer(model=dt2.trees[i], refset=X_train_np, verbose=False)
            tree_exp_shap_val = tree_exp.explain(foreground_sample)
            shap_sum += tree_exp_shap_val.sum()

        time_diff = datetime.now() - start_time
        
        print(f"Time taken = {time_diff}")

        raw_score = dt2.predict_raw(X_test_np)[foreground_index]

        self.assertTrue(np.isclose(shap_sum, raw_score, rtol=1e-2))

    # @unittest.skip
    def test_local_accuracy_half_vectorized(self):
        rows = 10000
        X_train_np, X_test_np, y_train_np, y_test_np, scale_pos_weight = self.load_data_and_scale_pos_weight(rows)

        max_depth, learning_rate, n_estimators, gamma, lb, base_score = self.load_parameter(max_depth=2)

        dt2 = XGClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=lb,
            min_child_weight=gamma,
            categorical_features=[i for i in range(X_train_np.shape[1])],
            base_score=base_score,
            verbose=False
        )

        dt2.fit(X_train_np, y_train_np)

        rng = np.random.default_rng()

        foreground_index = rng.choice(1000, 500, False)
        foreground_sample = X_test_np[foreground_index]

        base_val = dt2.predict_raw(X_train_np).mean()
        shap_sum = np.full(shape=len(foreground_sample), dtype=np.float32, fill_value=base_val)

        start_time = datetime.now()

        for i in range(n_estimators):
            tree_exp = IndependentTreeExplainer(model=dt2.trees[i], refset=X_train_np, verbose=False)
            tree_exp_shap_val = tree_exp.half_vectorized_explain(foreground_sample)
            shap_sum += tree_exp_shap_val.sum(axis=1)

        time_diff = datetime.now() - start_time
        
        print(f"Time taken = {time_diff}")

        raw_score = dt2.predict_raw(X_test_np)[foreground_index]

        print(shap_sum[:40])
        print("====")
        print(raw_score[:40])

        for t in np.isclose(shap_sum, raw_score, rtol=1e-2):
            self.assertEqual(t, np.True_)

if __name__ == '__main__':
    unittest.main()