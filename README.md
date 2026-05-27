# 💳 Credit Card Fraud Detection — XGBoost + SHAP

> Detecting fraudulent credit card transactions using XGBoost and SHAP explainability — achieving 0.9754 ROC AUC and 84% fraud recall on 284,807 real transactions with an estimated £58M annual saving potential.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![Dataset](https://img.shields.io/badge/Dataset-MLG--ULB-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## 📌 Overview

Credit card fraud costs the global economy billions annually. The challenge is extreme — only **0.172% of transactions are fraudulent**, making this one of the most severe class imbalance problems in machine learning. A naive model predicting "legitimate" for everything would be 99.83% accurate but completely useless.

This project builds a production-ready fraud detection system that not only achieves near-perfect detection performance but also **explains why each transaction was flagged** — critical for compliance, regulatory requirements, and analyst trust.

---

## 🎯 Key Results

| Metric | Score |
|---|---|
| ROC AUC | **0.9754** |
| Fraud Recall | **84%** |
| Fraud Precision | **77%** |
| Accuracy | **99.97%** |
| High Risk Precision | **84.4%** |
| Fraud rate in dataset | **0.172%** |

---

## 💰 Business Impact

| Metric | Value |
|---|---|
| Transactions analysed | 284,807 |
| High risk flags raised | 96 per 57k transactions |
| Confirmed fraud caught | 81 cases |
| False alarms | 15 cases |
| Net saving per 57k transactions | ~£9,149 |
| **Estimated annual saving (1M txns/day)** | **~£58,400,000** |

---

## 🚨 The Core Challenge — Extreme Class Imbalance

| Class | Count | Percentage |
|---|---|---|
| Legitimate | 284,315 | 99.828% |
| Fraud | 492 | 0.172% |

**Solution:** `scale_pos_weight = 577` — tells XGBoost to weight fraud cases 577x more important during training. This is more effective than SMOTE on a dataset of this size.

---

## 🔑 Top Fraud Indicators (SHAP Analysis)

| Rank | Feature | Signal |
|---|---|---|
| 1 | V14 | Transaction pattern deviation — strongest fraud signal |
| 2 | V12 | Merchant category anomaly |
| 3 | V4 | Cardholder behaviour deviation |
| 4 | V10 | Time-based pattern anomaly |
| 5 | V11 | Geographic/location signal |

*Note: V1-V28 are PCA-transformed features anonymised for privacy*

---

## ⚙️ Fraud Risk Tiers

| Risk Level | Probability | Action | Count |
|---|---|---|---|
| 🔴 High Risk | > 70% | Immediate human review | 96 |
| 🟡 Medium Risk | 30-70% | Analyst review | 42 |
| 🟢 Low Risk | < 30% | Auto-approve | 56,314 |

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **ML Model:** XGBoost Classifier
- **Explainability:** SHAP (TreeExplainer)
- **Imbalance Handling:** scale_pos_weight (577:1)
- **Preprocessing:** StandardScaler, Stratified Train/Test Split
- **Evaluation:** ROC AUC, Precision, Recall, F1
- **Visualisation:** Matplotlib, Seaborn, SHAP plots
- **Environment:** Jupyter Notebook

---

## 🔍 Methodology

```
Credit Card Dataset (284,807 transactions)
              ↓
Exploratory Data Analysis
(class imbalance, amount & time distributions)
              ↓
Preprocessing
(StandardScaler on Amount & Time, stratified split)
              ↓
XGBoost Classifier
(scale_pos_weight=577 for imbalance handling)
              ↓
SHAP Explainability
(global feature importance + directional impact)
              ↓
Risk Scoring System
(Low / Medium / High risk tiers)
              ↓
Business Impact Analysis
(£58M annual saving estimate)
```

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.ipynb     # Main analysis notebook
├── README.md                 # Project documentation
```

> **Note:** The dataset (150MB) is not included due to size limits.
> Download from: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/Anurag101723/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Download dataset from Kaggle and place `creditcard.csv` in the folder

3. Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
```

4. Open the notebook
```bash
jupyter notebook fraud_detection.ipynb
```

---

## 👤 Author

**Anurag Rathore**
M.Sc. Big Data & Business Analytics — FOM University of Applied Sciences
📧 anuragakrathore@gmail.com
🔗 [LinkedIn](https://linkedin.com/in/anurag1017) · [Portfolio](https://Anurag101723.github.io)
