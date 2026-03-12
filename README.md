
# Customer Churn Prediction: End-to-End MLOps Pipeline

A production-grade machine learning system for predicting customer churn on the IBM Telco dataset. This project covers the full ML lifecycle: exploratory analysis, feature engineering, multi-model training, hyperparameter optimization, model explainability, threshold tuning, experiment tracking, and automated drift monitoring.

The emphasis is on the full system, not just the model. The notebook documents every design decision and explains what breaks in production when each step is skipped.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Key Design Decisions](#key-design-decisions)
- [Production Architecture](#production-architecture)

---

## Problem Statement

A telecom company wants to identify customers likely to cancel their subscription before they do, so the retention team can intervene with targeted offers. The business constraint is a fixed retention budget. The model must rank customers by churn risk so that outreach is allocated to the customers most likely to leave.

This has two technical implications. First, ranking quality matters more than raw accuracy, making AUC-ROC and Average Precision the primary evaluation metrics. Second, the decision threshold is a business parameter, not a modeling default. It is determined by how many customers the retention team can contact per week, not by an arbitrary 0.5 cutoff.

---

## Dataset

IBM Telco Customer Churn, available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

- 7,043 customer records
- 20 raw features covering demographics, account information, and subscribed services
- Binary target: Churn (Yes / No)
- Class distribution: approximately 73% retained, 27% churned

The notebook includes an automatic fallback that generates a synthetic dataset with an identical schema if the Kaggle file is not present, so the full pipeline runs in any environment without manual setup.

---

## Project Structure

```
churn-prediction-mlops/
|
+-- churn_prediction_mlops.ipynb    # Main notebook -- full pipeline
+-- README.md
+-- requirements.txt
+-- .gitignore
|
+-- mlruns/                         # MLflow tracking directory (auto-generated on first run)
```

---

## Pipeline Overview

The notebook is structured as 15 sequential sections. Each section opens with a written explanation covering what the step does, why it exists, what it enables downstream, and what breaks in production if it is omitted.

**Section 0 -- Environment and Reproducibility**
Seeds NumPy, Python random, and the environment hash. Initializes MLflow experiment tracking so every run is logged automatically from the start.

**Section 1 -- Data Loading and Schema Audit**
Loads the Telco dataset. Audits column types, null counts, cardinality, and target distribution before any transformations are applied.

**Section 2 -- Data Cleaning and Type Correction**
Resolves the TotalCharges type bug caused by whitespace-as-null values. Collapses hierarchical "No internet service" and "No phone service" strings into binary flags rather than treating them as independent categories. One-hot encodes remaining categoricals.

**Section 3 -- Exploratory Data Analysis**
Quantifies churn rate by contract type, tenure cohort, and monthly spend. Documents why class imbalance makes accuracy a misleading metric and establishes AUC and Average Precision as the correct evaluation frame.

**Section 4 -- Feature Engineering**
Constructs five engineered features with documented business rationale: charges per tenure month (value density), total service count (switching cost proxy), high-risk segment flag for month-to-month customers under 12 months tenure, tenure bucket for nonlinear tenure encoding, and log-transformed total charges to correct right-skew.

**Section 5 -- Train/Test Split**
Stratified 80/20 split performed before any fitting, including scaling. Scaling parameters are fit on the training set only and applied to the test set to prevent leakage.

**Section 6 -- Logistic Regression Baseline**
Establishes a performance floor using a regularized logistic regression. Evaluates with AUC-ROC, Average Precision, Brier Score, and Log Loss. Documents why a baseline is required before building more complex models.

**Section 7 -- XGBoost**
Trains a gradient boosted tree classifier with class weighting, L1/L2 regularization, column subsampling, and early stopping on a held-out validation split.

**Section 8 -- LightGBM**
Trains a leaf-wise gradient boosted classifier. Explains the architectural differences from XGBoost and why both models are worth running: they make partially uncorrelated errors, which makes ensembling effective.

**Section 9 -- Hyperparameter Optimization with Optuna**
Uses Tree-structured Parzen Estimator (TPE) Bayesian optimization over 50 trials, evaluated on stratified 5-fold cross-validation. MedianPruner terminates unpromising trials early to reduce total compute.

**Section 10 -- Model Comparison**
Plots ROC curves and Precision-Recall curves for all five models including the ensemble. Explains why the Precision-Recall curve is the more informative metric for imbalanced classification and includes a tabular summary of all evaluation metrics.

**Section 11 -- SHAP Explainability**
Uses TreeExplainer to compute exact SHAP values for all test samples. Produces a global feature importance bar chart, a beeswarm plot showing directional impact, and a per-customer explanation showing the top drivers of a specific high-risk prediction.

**Section 12 -- Threshold Optimization**
Computes F1 score, precision, and recall across all thresholds from 0.1 to 0.9. Identifies the threshold that maximizes F1 and explains how to replace it with a business-cost-driven threshold when false positive and false negative costs are known.

**Section 13 -- Drift Detection**
Implements data drift monitoring using Evidently AI, with a Population Stability Index fallback for offline environments. Simulates a distribution shift by perturbing MonthlyCharges and tenure to demonstrate how the monitoring layer catches it.

**Section 14 -- MLflow Model Registry**
Logs the final tuned XGBoost model with full parameter and metric metadata to the MLflow Model Registry. Documents how to load the registered model artifact in a serving layer.

**Section 15 -- Summary and Production Readiness Checklist**
Documents every design decision made across the pipeline, the rationale behind each, and the next steps required to move from this notebook to a deployed production service.

---

## Results

| Model | AUC-ROC | Average Precision | Brier Score |
|---|---|---|---|
| Logistic Regression | ~0.840 | ~0.630 | ~0.150 |
| XGBoost (default) | ~0.860 | ~0.660 | ~0.140 |
| LightGBM | ~0.862 | ~0.663 | ~0.138 |
| XGBoost (Optuna-tuned) | ~0.870 | ~0.675 | ~0.133 |
| Ensemble (XGB + LGBM) | ~0.873 | ~0.680 | ~0.131 |

Results are approximate and will vary slightly depending on the dataset version and environment. All runs are tracked in MLflow with exact values logged per experiment.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | pandas, NumPy |
| Modeling | scikit-learn, XGBoost, LightGBM |
| Hyperparameter optimization | Optuna |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| Drift monitoring | Evidently AI |
| Visualization | matplotlib, seaborn |

---

## Setup and Installation

Clone the repository:

```bash
git clone https://github.com/SreeTatikonda/churn-prediction-mlops.git
cd churn-prediction-mlops
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # macOS and Linux
venv\Scripts\activate           # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Contents of requirements.txt:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
optuna
shap
mlflow
evidently
jupyter
```

Launch the notebook:

```bash
jupyter notebook churn_prediction_mlops.ipynb
```

View MLflow experiment runs:

```bash
mlflow ui
```

Open http://localhost:5000 in a browser to see all logged runs, parameters, and metrics.

---

## Key Design Decisions

**Why AUC-ROC and Average Precision instead of accuracy**

The dataset has a 27% churn rate. A model predicting no churn for every customer achieves 73% accuracy while being completely useless. AUC-ROC measures ranking quality independently of threshold. Average Precision is more informative than AUC-ROC on imbalanced datasets because it penalizes false positives more heavily and does not benefit from the large true-negative pool.

**Why class weighting instead of SMOTE**

SMOTE generates synthetic samples by interpolating between minority class observations. On tabular data with mixed binary, categorical, and continuous features, synthetic interpolation produces unrealistic records. Class weighting adjusts the loss function to penalize minority class errors more heavily, achieving the same effect without introducing artificial data artifacts.

**Why Optuna over grid search**

Grid search evaluates every combination in a predefined space, which grows exponentially with the number of parameters. Optuna's TPE algorithm models the distribution of high-performing configurations and proposes the next trial based on that model. Over 50 trials it converges on near-optimal configurations far more efficiently than grid or random search.

**Why SHAP over built-in feature importances**

Built-in feature importances for tree models produce a single global ranking. SHAP produces an importance value per feature per prediction, enabling per-customer explanations. This is what makes the model actionable for a retention team: not just "contract type matters" but "for this specific customer, being on a month-to-month contract is adding 0.18 to their churn probability."

**Why the decision threshold is not 0.5**

A threshold of 0.5 implicitly assumes equal cost for false positives and false negatives. In a retention context, a false negative (missing a churner) costs the customer's lifetime value. A false positive (flagging a loyal customer) costs a retention offer. These costs are not equal. The notebook computes the F1-optimal threshold and documents how to derive a cost-weighted threshold when the business can quantify the cost ratio explicitly.

**Why drift monitoring is included**

A model trained on historical data degrades as the world changes. Monthly charges shift when pricing changes. Tenure distributions shift when acquisition slows. Contract mixes shift after promotional campaigns. Without monitoring, degradation is silent: the model continues returning confident predictions while its accuracy quietly falls. The drift detection layer is what separates a model that works from a model that works reliably over time.

---

## Production Architecture

The notebook is the research and validation layer. A complete production deployment adds the following components.

**Serving layer**
A FastAPI endpoint that accepts a customer record as JSON, applies the logged scaler parameters, runs inference through the registered MLflow model, and returns the churn probability, the binary prediction at the optimized threshold, and the top three SHAP-derived reasons driving the prediction.

**Containerization**
The FastAPI service and model artifact packaged in a Docker container for consistent deployment across environments.

**Orchestration**
An Airflow DAG that runs weekly: pulls recent inference logs, computes PSI and Evidently drift metrics against the training distribution, and triggers a retraining job if drift exceeds the configured threshold.

**Monitoring dashboard**
A Streamlit application showing the current at-risk customer list, per-customer SHAP explanations, model performance over time, and drift metric history -- designed for a retention team manager, not a data scientist.

---

## Author

Yasaswini Tatikonda

[Portfolio](https://yasaswinitatikonda.netlify.app) | [GitHub](https://github.com/SreeTatikonda) | [Medium](https://medium.com/@yasaswinitatikonda)
