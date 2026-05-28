# Credit Card Fraud Detection

XGBoost classifier for fraud detection on 284,807 real credit card transactions with SHAP explainability and a three-tier risk scoring system.

---

## Overview

Credit card fraud detection is one of the most severe class imbalance problems in applied machine learning. Only 0.172% of transactions in this dataset are fraudulent, meaning a naive model predicting legitimate for every transaction would achieve 99.83% accuracy while being completely useless.

This project builds a production-ready fraud detection system that handles extreme class imbalance using scale_pos_weight, achieves near-perfect ROC AUC, explains why each transaction was flagged using SHAP, and categorises transactions into actionable risk tiers for compliance teams.

---

## Results

| Metric | Score |
|---|---|
| Model | XGBoost Classifier |
| ROC AUC | 0.9754 |
| Fraud Recall | 84% |
| Fraud Precision | 77% |
| Accuracy | 99.97% |
| High Risk Tier Precision | 84.4% |

---

## Dataset

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Legitimate | 284,315 (99.828%) |
| Fraudulent | 492 (0.172%) |
| Features | 30 (V1-V28 PCA transformed, Amount, Time) |
| Source | MLG-ULB, Kaggle |

Note: V1 through V28 are PCA-transformed features anonymised for privacy. Only Amount and Time are original columns.

---

## Class Imbalance Handling

Rather than SMOTE oversampling, which is computationally prohibitive at this scale, scale_pos_weight was used to weight fraud cases during training.

| Parameter | Value | Meaning |
|---|---|---|
| scale_pos_weight | 577 | Fraud cases weighted 577x more important |
| Ratio | 577:1 | Legitimate to fraud in training set |

---

## Risk Scoring System

| Risk Level | Probability Threshold | Action | Count in Test Set |
|---|---|---|---|
| High Risk | Above 70% | Immediate human review | 96 |
| Medium Risk | 30% to 70% | Analyst review | 42 |
| Low Risk | Below 30% | Auto-approve | 56,314 |

Of the 96 high-risk flags, 81 were confirmed fraud — a precision of 84.4%.

---

## Top Fraud Indicators (SHAP)

| Rank | Feature | Signal |
|---|---|---|
| 1 | V14 | Strongest fraud indicator — transaction pattern deviation |
| 2 | V12 | Merchant category anomaly |
| 3 | V4 | Cardholder behaviour deviation |
| 4 | V10 | Time-based pattern anomaly |
| 5 | V11 | Geographic or location signal |

---

## Business Impact

| Metric | Value |
|---|---|
| Fraud cases caught per 57k transactions | 81 |
| False alarms | 15 |
| Missed fraud cases | 17 |
| Average fraud transaction value | 122.21 GBP |
| Net saving per 57k transactions | ~9,149 GBP |
| Estimated annual saving at 1M transactions per day | ~58,400,000 GBP |

---

## Operational Recommendations

| Score Range | Action |
|---|---|
| Above 95% | Auto-block — model is near certain |
| 70% to 95% | Human review by compliance analyst |
| 30% to 70% | Flag for monitoring |
| Below 30% | Auto-approve |

Retrain model monthly. Fraud patterns evolve and model performance will degrade without regular updates. Monitor V14 and V12 signals as early warning indicators.

---

## Methodology

```
Credit Card Dataset (284,807 transactions)

Exploratory Data Analysis
Class imbalance, amount distribution, time patterns

Preprocessing
StandardScaler on Amount and Time
V1-V28 already scaled from PCA
Stratified 80/20 train-test split

XGBoost Training
scale_pos_weight = 577 to handle class imbalance
n_estimators = 200, max_depth = 6, learning_rate = 0.05

SHAP Analysis
TreeExplainer on 1,000 test samples
Global feature importance and directional impact

Risk Scoring
Three-tier system: Low, Medium, High
Business impact quantification
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML Model | XGBoost Classifier |
| Explainability | SHAP (TreeExplainer) |
| Preprocessing | StandardScaler, Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Repository Structure

```
credit-card-fraud-detection/
|
|-- fraud_detection.ipynb
|-- README.md
```

Dataset not included due to size (150MB).
Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## How to Run

```bash
git clone https://github.com/Anurag101723/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
jupyter notebook fraud_detection.ipynb
```

---

## Author

Anurag Rathore  
anuragakrathore@gmail.com  
linkedin.com/in/anurag1017  
anurag101723.github.io
