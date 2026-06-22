# Predictive Maintenance with Contextual Data Fusion

![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=for-the-badge)
![Development Stage](https://img.shields.io/badge/Development%20Stage-Week%202-blue?style=for-the-badge)

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

## Rolling Statistics Features

To capture operational trends, rolling window features were generated using a 10-observation window.

Generated features:

- Rolling Mean
- Rolling Standard Deviation
- Rolling Variance

for all primary telemetry signals.

## Lag Features

Historical telemetry information was incorporated through lag features.

Generated:

- Lag 1
- Lag 2

for all primary sensor signals, enabling the model to leverage previous machine states.

## Trend Features

To capture temporal dynamics, change features were generated for all primary telemetry signals.

Generated features:

- Air Temperature Change
- Process Temperature Change
- Rotational Speed Change
- Torque Change
- Tool Wear Change

These features quantify short-term operational shifts and provide additional predictive signals.

## Telemetry Feature Engineering

Generated feature groups:

| Feature Type | Count |
|-------------|-------:|
| Raw Features | 14 |
| Rolling Features | 15 |
| Lag Features | 10 |
| Change Features | 5 |

Total Features: 44

The validated telemetry dataset serves as the foundation for contextual feature integration.

## Contextual Data Integration

To simulate real-world operating conditions, contextual variables were introduced:

| Feature | Description |
|----------|------------|
| Ambient Humidity | Environmental condition |
| Energy Load Index | Operational load estimate |
| Production Demand | Simulated production pressure |
| Days Since Maintenance | Maintenance recency |
| Shift | Day/Night operation |

These variables serve as external context signals for predictive maintenance modeling.

## Contextual Data Fusion

Interaction features were generated to combine telemetry signals with operational context.

Generated Features:

| Interaction Feature | Description |
|---------------------|-------------|
| Torque × Load | Mechanical stress under load |
| Wear × Demand | Tool degradation under production pressure |
| Temperature × Humidity | Thermal-environment interaction |

These features provide richer representations of machine operating conditions.

### Context Fusion Validation

Validation confirmed:

- No missing values
- 5 contextual features
- 3 interaction features
- 52 total columns

The dataset is now ready for comparative modeling and ablation analysis.

## Ablation Study Design

Two predictive maintenance configurations are evaluated:

| Configuration | Features |
|--------------|----------|
| Telemetry Only | Engineered telemetry features |
| Telemetry + Context | Telemetry + contextual data fusion features |

The objective is to quantify the contribution of contextual information to predictive performance.