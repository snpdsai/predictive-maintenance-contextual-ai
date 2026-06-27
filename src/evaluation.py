import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    confusion_matrix,
)

from src.config import NOISE_LEVELS, THRESHOLDS


def add_noise(data, noise_level):

    noisy = data.copy()

    sensor_cols = [
        "Air_temperature__K_",
        "Process_temperature__K_",
        "Rotational_speed__rpm_",
        "Torque__Nm_",
        "Tool_wear__min_",
    ]

    for col in sensor_cols:

        if col in noisy.columns:

            noise = np.random.normal(0, noise_level * noisy[col].std(), len(noisy))

            noisy[col] += noise

    return noisy


def noise_analysis(model, X_test, y_test):

    noise_levels = NOISE_LEVELS

    results = []

    for level in noise_levels:

        X_noisy = add_noise(X_test, level)

        preds = model.predict(X_noisy)

        score = f1_score(y_test, preds, average="macro")

        results.append([level, score])

    return pd.DataFrame(results, columns=["Noise_Level", "Macro_F1"])


def threshold_tuning(y_true, y_prob, thresholds=None):

    if thresholds is None:
        thresholds = THRESHOLDS

    results = []

    for t in thresholds:

        preds = (y_prob >= t).astype(int)

        results.append(
            [
                t,
                precision_score(y_true, preds, zero_division=0),
                recall_score(y_true, preds, zero_division=0),
                f1_score(y_true, preds, zero_division=0),
            ]
        )

    return pd.DataFrame(results, columns=["Threshold", "Precision", "Recall", "F1"])


def evaluate_holdout(model, X_test, y_test):

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    return {
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "average_precision": average_precision_score(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds),
    }
