import pandas as pd
import numpy as np
import re

from src.config import SENSOR_COLS


def create_telemetry_features(df, window_size=10):

    df = df.copy()

    # Rolling Mean
    for col in SENSOR_COLS:
        df[f"{col}_roll_mean"] = (
            df[col]
            .rolling(window=window_size)
            .mean()
        )

    # Rolling Std
    for col in SENSOR_COLS:
        df[f"{col}_roll_std"] = (
            df[col]
            .rolling(window=window_size)
            .std()
        )

    # Rolling Variance
    for col in SENSOR_COLS:
        df[f"{col}_roll_var"] = (
            df[col]
            .rolling(window=window_size)
            .var()
        )

    # Lag Features
    for col in SENSOR_COLS:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)

    # Temperature Difference
    df["temp_difference"] = (
        df["Process temperature [K]"]
        - df["Air temperature [K]"]
    )

    # Rate of Change
    for col in SENSOR_COLS:
        df[f"{col}_change"] = df[col].diff()

    df = df.dropna().reset_index(drop=True)

    return df


def add_context_features(df):

    np.random.seed(42)

    df = df.copy()

    df["ambient_humidity"] = np.random.normal(
        60, 10, len(df)
    ).clip(30, 90)

    df["energy_load_index"] = np.random.normal(
        70, 15, len(df)
    ).clip(20, 100)

    df["production_demand"] = np.random.normal(
        75, 12, len(df)
    ).clip(30, 100)

    df["days_since_maintenance"] = np.random.randint(
        1, 181, len(df)
    )

    df["shift"] = np.random.choice(
        [0, 1],
        size=len(df)
    )

    return df


def create_interaction_features(df):

    df = df.copy()

    df["torque_x_load"] = (
        df["Torque [Nm]"]
        * df["energy_load_index"]
    )

    df["wear_x_demand"] = (
        df["Tool wear [min]"]
        * df["production_demand"]
    )

    df["temp_x_humidity"] = (
        df["Process temperature [K]"]
        * df["ambient_humidity"]
    )

    return df


def clean_column_names(df):

    df.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", col)
        for col in df.columns
    ]

    return df