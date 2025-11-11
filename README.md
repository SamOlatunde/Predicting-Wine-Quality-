# Wine Quality Classification 

This project develops and compares multiple **supervised machine learning models** to predict wine quality based on physicochemical properties. It explores how feature selection, hyperparameter tuning, and imbalance handling techniques affect predictive performance and interpretability.

---

## Objectives

* Build, tune, and evaluate multiple models for wine quality prediction.
* Handle class imbalance using **SMOTE**, **undersampling**, and **class weighting**.
* Analyze **feature importance** using **Backward Feature Selection** and **Permutation Importance**.
* Compare models using multiple metrics to determine the best-performing and most interpretable approach.

---

## Models Implemented

* **Logistic Regression** (L1, L2, weighted, SMOTE, undersampled)
* **K-Nearest Neighbors (KNN)** (with/without SMOTE and undersampling)
* **Decision Tree** (with/without SMOTE)
* **Random Forest** (standard, balanced, and SMOTE)
* **Support Vector Machine (SVM)** (standard and SMOTE)

---

## Dataset

* **Source:** [UCI Machine Learning Repository – Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* **Samples:** ~5,000
* **Features:** 11 numerical physicochemical properties
* **Target:** Wine quality (integer scores from 1–10)

---

## Workflow Overview

1. **Data Loading & Exploration**

   * Analyze class distribution and detect imbalance.
2. **Data Preprocessing**

   * Split (70/15/15), scale with `StandardScaler`.
3. **Model Training & Tuning**

   * GridSearchCV with cross-validation for hyperparameter optimization.
   * Manual hyperparameter optimization for 
4. **Imbalance Handling**

   * SMOTE, Random Undersampling, or class weighting.
5. **Feature Selection**

   * Backward elimination or Permutation Importance (threshold = {0.01,0.25 ).
6. **Evaluation**

   * Metrics: Accuracy, Precision, Recall, Macro & Weighted F1-score.
   * Results summarized in the report.

---

## Results Summary

* **Best Overall Model:** K-Nearest Neighbors (no resampling)
* **Key Features:** *alcohol*, *volatile_acidity*, *chlorides*, *ph*, *volatile_acidity*, *free_sulfur_dioxide*, *total_sulfur_dioxide*
* **Best Validation Metrics:** Accuracy ≈ 0.654, F1 ≈ 0.348



---

## Tools & Libraries

* **Python 3.12+**
* **scikit-learn** – model training and evaluation
* **imbalanced-learn** – SMOTE and Balanced Random Forest
* **pandas, numpy** – data processing
* **matplotlib, seaborn** – visualization

---

## Repository Structure

```
📁 wine-quality-ml
│
├── notebooks/
│   ├── logistic_regression.ipynb
│   ├── knn.ipynb
│   ├── decision_tree.ipynb
│   ├── random_forest.ipynb
│   └── svm.ipynb
├── report/
│   └── Wine_Quality_Report.pdf
└── README.md
```

---

## Author

**Samuel Olatunde**
Midwestern State University
---
