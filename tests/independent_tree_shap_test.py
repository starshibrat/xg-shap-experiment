from mini_shap.independent_tree_shap import IndependentTreeExplainer
import pandas as pd
import unittest
from tree.xg_classifier import XGClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime

class TestTreeSHAP(unittest.TestCase):
    def test_local_accuracy_single(self):
        current_dir = Path(__file__).parent
        file_path = current_dir / "../datasets/BRFSS/out_v2.parquet"
        df = pd.read_parquet(file_path.resolve())
        limit = 50000

        y = df.iloc[:limit, 0]
        X = df.iloc[:limit, 1:].astype("category")

        X_np = X.to_numpy().astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

        X_train_np, X_test_np, y_train_np, _ = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32)

        classes = y_train.value_counts()
        max_depth = 1
        learning_rate = 0.1
        n_estimators = 3
        gamma = 0
        lb = 1
        scale_pos_weight = classes[0]/classes[1]
        base_score = 0.5

        dt2 = XGClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            reg_lambda=lb,
            min_child_weight=gamma,
            categorical_features=[i for i in range(len(X.columns))],
            base_score=base_score,
            verbose=False
        )

        dt2.fit(X_train_np, y_train_np)

        foreground_index = 400
        foreground_sample = X_test_np[foreground_index]

        base_val = dt2.predict_raw(X_np).mean()
        shap_sum = base_val

        start_time = datetime.now()

        for i in range(n_estimators):
            tree_exp = IndependentTreeExplainer(model=dt2.trees[i], refset=X_np, verbose=False)
            tree_exp_shap_val = tree_exp.explain(foreground_sample)
            shap_sum += tree_exp_shap_val.sum()

        time_diff = datetime.now() - start_time
        
        print(f"Time taken = {time_diff}")

        raw_score = dt2.predict_raw(X_test_np)[foreground_index]

        self.assertTrue(np.isclose(shap_sum, raw_score, rtol=1e-2))
        

if __name__ == '__main__':
    unittest.main()