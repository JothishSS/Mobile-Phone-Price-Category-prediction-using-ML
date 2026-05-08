# Mobile Phone Price Range Prediction Using Machine Learning

## Project Overview

This project predicts the price category of mobile phones based on their hardware specifications using machine learning.

The project explores:
- Multinomial Logistic Regression
- Random Forest
- XGBoost
- Ordinal Logistic Regression

The final model selected was Ordinal Logistic Regression due to its superior accuracy and alignment with the ordered nature of the target variable.

---

## Problem Statement

Predict mobile phone price categories:

| Label | Category |
|------|-----------|
| 0 | Low Cost |
| 1 | Medium Cost |
| 2 | High Cost |
| 3 | Very High Cost |

---

## Dataset Information

- Rows: 2000
- Features: 20
- Target Variable: price_range

Key features include:
- RAM
- Battery Power
- Display Resolution
- Internal Memory
- Processor Specifications

---

## Models Evaluated

| Model | Accuracy |
|------|------------|
| Logistic Regression | 96.5% |
| Random Forest | 88.0% |
| XGBoost | 92.5% |
| Ordinal Logistic Regression | 98.0% |

---

## Key Findings

- RAM was the most influential feature.
- Logistic Regression performed strongly due to linear separability.
- Random Forest underperformed because of fragmented decision boundaries.
- Ordinal Logistic Regression achieved the best performance by modeling class hierarchy.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- MORD
- Matplotlib
- Seaborn

---

## Project Structure

```text
├── mobile_price_prediction.py
├── README.md
├── requirements.txt
├── model_accuracy_comparison.png
├── confusion_matrix.png
└── mobile_price_classification.csv