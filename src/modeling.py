import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score

from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier

import joblib

from src.config import LIGHTGBM_PARAMS, N_SPLITS, RANDOM_STATE

DROP_COLS = ["UDI", "Product_ID", "Machine_failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


def prepare_data(df):

    df = df.copy()

    df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    X = df.drop(columns=DROP_COLS)

    y = df["Machine_failure"]

    return X, y


def cross_validate_model(X, y):

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = []

    for train_idx, val_idx in skf.split(X, y):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        smote = SMOTE(random_state=42)

        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = LGBMClassifier(**LIGHTGBM_PARAMS)

        model.fit(X_train_res, y_train_res)

        preds = model.predict(X_val)

        score = f1_score(y_val, preds, average="macro")

        scores.append(score)

    return scores


def train_final_model(X, y):

    smote = SMOTE(random_state=42)

    X_resampled, y_resampled = smote.fit_resample(X, y)

    model = LGBMClassifier(**LIGHTGBM_PARAMS)

    model.fit(X_resampled, y_resampled)

    return model


def save_model(model, path):

    joblib.dump(model, path)

    print(f"Model saved -> {path}")
