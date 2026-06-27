# src/config.py

# Random Seed

RANDOM_STATE = 42

# Data Paths

RAW_DATA_PATH = "../data/raw/ai4i2020.csv"

TELEMETRY_DATA_PATH = (
    "../data/processed/telemetry_features.csv"
)

CONTEXT_DATA_PATH = (
    "../data/processed/context_fused_dataset.csv"
)

MODEL_PATH = (
    "../models/lightgbm_predictive_maintenance_final.pkl"
)

# Cross Validation

N_SPLITS = 5

# SMOTE

SMOTE_RANDOM_STATE = 42

# LightGBM Parameters

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": RANDOM_STATE
}

# Noise Analysis

NOISE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20
]

# Threshold Tuning

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70
]

# Sensor Columns

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]