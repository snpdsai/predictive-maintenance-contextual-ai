# src/utils.py

import pandas as pd
import matplotlib.pyplot as plt


def print_shape(df, name="DataFrame"):
    print(f"{name} Shape: {df.shape}")


def save_dataframe(df, path):

    df.to_csv(path, index=False)

    print(f"Saved -> {path}")


def plot_feature_importance(
    importance_df,
    top_n=15
):

    top_features = (
        importance_df
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        f"Top {top_n} Feature Importances"
    )

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.show()


def plot_noise_analysis(noise_df):

    plt.figure(figsize=(8,5))

    plt.plot(
        noise_df["Noise_Level"],
        noise_df["Macro_F1"],
        marker="o"
    )

    plt.xlabel("Noise Level")
    plt.ylabel("Macro F1")

    plt.title(
        "Noise Robustness Analysis"
    )

    plt.grid(True)

    plt.show()


def plot_threshold_tuning(
    threshold_df
):

    plt.figure(figsize=(8,5))

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        marker="o",
        label="Precision"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        marker="o",
        label="Recall"
    )

    plt.plot(
        threshold_df["Threshold"],
        threshold_df["F1"],
        marker="o",
        label="F1"
    )

    plt.legend()

    plt.grid(True)

    plt.title(
        "Threshold Tuning"
    )

    plt.show()