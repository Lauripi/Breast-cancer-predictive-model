# Breast Cancer Classification - Model Building, Predictor Selection and Validation - Python

A complete, leakage-aware machine learning pipeline for classifying breast tumors as malignant or benign. It is built not just to chase accuracy, but to produce a model whose feature selection and decision threshold can actually be defended and explained.

## Overview

- **Dataset:** Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 morphological features extracted from digitized biopsy images)
- **Task:** binary classification — malignant vs. benign tissue
- **Focus:** methodological rigor over a single headline number: handling multicollinearity properly, validating feature selection stability, and choosing a decision threshold aligned with clinical priorities

## Results at a glance

| Metric | Value |
|---|---|
| ROC-AUC | 0.996 |
| Accuracy | 97.4% |
| Sensitivity (malignant recall) | 95.2% |
| Specificity | 98.6% |
| Features used | 12 (reduced from 30) |

## Approach

1. **Exploratory analysis and correlation screening:** identified strong multicollinearity among size-related features (radius, perimeter, area are mathematically redundant); manually removed 9 redundant features before modeling, using training data only.\
   → *Multicollinearity diagnosis, correlation-based feature reduction, leakage-safe workflow*
2. **Model comparison:** benchmarked 6 classifiers (Logistic Regression, KNN, Linear/RBF SVM, Random Forest, Gradient Boosting) via 10-fold stratified cross-validation, with scaling fit within pipeline folds. Logistic Regression was selected for its interpretability, at no measurable cost in performance.\
   → *Systematic model benchmarking, cross-validation, scikit-learn pipelines*
3. **Regularization comparison:** compared L1, L2 and Elastic Net penalties; L1 produced the sparsest model (12 features) with no loss in ROC-AUC.\
   → *Regularized logistic regression, hyperparameter tuning (GridSearchCV), nested cross-validation*
4. **Threshold optimization:** compared a balanced (Youden) threshold against a minimum-sensitivity target, and selected the final threshold based on its best compromise between sensitivity and specificity. Chosen from training CV, then applied once to a held-out test set.\
   → *Clinical decision-analytic thinking, sensitivity/specificity trade-offs, cross-validated vs. held-out performance*
5. **Feature stability validation:** bootstrap-resampled the L1 feature selection (200 iterations) to separate genuinely robust predictors from features that only appeared due to one particular data split.\
   → *Bootstrap stability analysis*

## Tech stack

Python · scikit-learn · pandas · NumPy · matplotlib

## Files

- `1. Predictive_model_on_breast_cancer.ipynb` — full analysis notebook
