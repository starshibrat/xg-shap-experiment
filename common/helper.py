from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import xgboost as xgb
from tree.xg_classifier import XGClassifier
from typing import Tuple

def load_brfss(
    dataset_version: str = None,
    columns: list[str] = None,
):
    if dataset_version is None:
        dataset_version = "v3"
    
    current_dir = Path(__file__).parent
    file_path = current_dir / f"../datasets/BRFSS/out_{dataset_version}.parquet"
    df = pd.read_parquet(file_path.resolve())

    if columns is None:
        columns = list(df.columns)
    else:
        if not ("DIABETES" in columns):
            columns.insert(0, "DIABETES")

    return df[columns]

def calculate_scale_pos_weight(
    y: Tuple[pd.Series, np.ndarray]
):
    pos_class, neg_class = 0, 0

    if isinstance(y, pd.Series):
        classes = y.value_counts()
        pos_class = classes[1]
        neg_class = classes[0]
    elif isinstance(y, np.ndarray):
        classes = np.bincount(y)
        pos_class = classes[1]
        neg_class = classes[0]

    return neg_class / pos_class

def load_data_and_scale_pos_weight(
        size: int = None,
        return_numpy: bool=False,
        cast_string=False,
        tsize: np.float32=0.2,
        rdstate: int=1,
        columns: list[str]=None
):
        df = load_brfss(columns=columns)

        if cast_string:
            cast_to_string_categorical_data(df)

        if size is None:
            size = len(df)

        y = df.iloc[:size, 0]
        X = df.iloc[:size, 1:].astype("category")

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rdstate, test_size=tsize, stratify=y)

        scale_pos_weight = calculate_scale_pos_weight(y_train)

        if not return_numpy:
            return X_train, X_test, y_train, y_test, scale_pos_weight

        X_train_np, X_test_np, y_train_np, y_test_np = X_train.to_numpy().astype(np.float32), X_test.to_numpy().astype(np.float32), y_train.to_numpy().astype(np.float32), y_test.to_numpy().astype(np.float32)

        return X_train_np, X_test_np, y_train_np, y_test_np, scale_pos_weight

def cast_to_string_categorical_data(df: pd.DataFrame):
    for col in df.columns:
        if col == "DIABETES":
            continue

        df[col] = df[col].astype(dtype="Int8").astype("str")

def load_dummy_xgb_parameter():
    base_score = 0.5
    objective = "binary:logistic"
    random_state = 1
    max_depth = 1
    n_estimators = 5
    eta = 0.1
    enable_categorical = True
    max_cat_to_one_hot = 1
    reg_lambda = 0.1
    min_child_weight = 0

    return base_score, objective, random_state, max_depth, n_estimators, eta, enable_categorical, max_cat_to_one_hot, reg_lambda, min_child_weight

def load_dummy_lib_classifier(scale_pos_weight: np.float32=1, md: int=None, ntree: int=None) -> xgb.XGBClassifier:
    base_score, objective, random_state, max_depth, n_estimators, eta, enable_categorical, max_cat_to_one_hot, reg_lambda, min_child_weight = load_dummy_xgb_parameter()

    if md is not None:
        max_depth = md
    
    if ntree is not None:
        n_estimators = ntree

    m = xgb.XGBClassifier(
        base_score=base_score,
        objective=objective,
        random_state=random_state,
        max_depth=max_depth,
        n_estimators=n_estimators,
        eta=eta,
        enable_categorical=enable_categorical,
        max_cat_to_one_hot=max_cat_to_one_hot,
        scale_pos_weight=scale_pos_weight,
        reg_lambda = reg_lambda,
        min_child_weight = min_child_weight
    )

    return m

def load_dummy_scratch_classifier(scale_pos_weight: np.float32=1, md: int=None) -> XGClassifier:
    base_score, objective, random_state, max_depth, n_estimators, eta, enable_categorical, max_cat_to_one_hot, reg_lambda, min_child_weight = load_dummy_xgb_parameter()

    if md is not None:
        max_depth = md

    m = XGClassifier(
        objective=objective,
        max_depth=max_depth,
        base_score=base_score,
        n_estimators=n_estimators,
        learning_rate=eta,
        categorical_features=[i for i in range(32)],
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight
    )

    return m

