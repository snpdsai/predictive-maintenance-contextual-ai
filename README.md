# Predictive Maintenance with Contextual Data Fusion

![Status](https://img.shields.io/badge/Status-Active%20Development-orange)
![Development Stage](https://img.shields.io/badge/Development%20Stage-Week%201-blue)

Industrial predictive maintenance system using IoT telemetry, contextual data fusion, SMOTE, and LightGBM.

## Dataset

This project uses the AI4I Predictive Maintenance Dataset, containing telemetry measurements from industrial equipment.

### Features

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear

### Target

- Machine Failure

## Data Quality Assessment

Initial quality checks revealed:

- No missing values
- No duplicate records
- Failure rate of approximately 3.39%
- Highly imbalanced target variable

The class imbalance motivates the use of SMOTE during model training.

## Exploratory Analysis

Exploratory visualization was performed to understand:

- Temperature distributions
- Rotational speed patterns
- Torque behavior
- Tool wear characteristics
- Feature correlations

Insights from this stage informed the subsequent feature engineering process.

## Failure Pattern Analysis

Machine failures were analyzed against key telemetry signals.

Observations indicated that tool wear, torque, and rotational speed exhibit meaningful differences between failure and non-failure observations, motivating subsequent feature engineering efforts.

## Telemetry Signal Preparation

The raw telemetry dataset was prepared for feature engineering by identifying core sensor signals and analyzing their statistical properties.

Primary signals include:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear