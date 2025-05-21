# 📊 The Battle of the Regressors: Linear vs Ridge vs Lasso

Welcome to a project that explores one of the most fundamental challenges in machine learning — **how to balance model complexity and generalization**.

This project was completed as part of my graduate coursework (AIT 636 – Machine Learning) at George Mason University. In it, I compare three popular linear models — **Ordinary Least Squares**, **Lasso Regression**, and **Ridge Regression** — to understand how they perform on structured numerical data.

---

## 🧠 What Are These Regression Models?

### 🔹 Linear Regression (Least Squares)
A simple model that fits a straight line to minimize the **sum of squared errors**. While powerful, it can overfit when too many features are involved.

### 🔹 Ridge Regression
Adds a penalty for large weights (L2 regularization), **shrinking coefficients** to prevent overfitting but keeps all features.

### 🔹 Lasso Regression
Adds a different penalty (L1 regularization), which can **shrink some weights to zero** — effectively selecting important features and ignoring the rest.

---

## 🧪 What This Project Covers

- Implemented and compared:
  - **Linear Regression**
  - **Ridge Regression**
  - **Lasso Regression**
- Trained models on structured numerical data
- Explored effects of **regularization strength (alpha)** on model weights
- Evaluated models using:
  - Mean Squared Error (MSE)
  - R² Score
  - Coefficient analysis
- Visualized and interpreted how regularization alters model behavior

---

## 🧬 Dataset

A numerical dataset with continuous features and a regression target. The goal was to **predict a continuous outcome** and observe how each regression technique handled complexity and multicollinearity.

---

## 🗂️ Project Files

- `ait636_linear,ridge,lasso_regression_assignment3c_py.py`: Python script implementing all three regression models
- `AIT_636_001_Assignment3C_SamhitaSarikonda.pdf`: Full report including analysis, results, and visualizations
- `README.md`: This file
- `requirements.txt`: Python dependencies used

---

## 🧰 Libraries Used

```bash
scikit-learn
numpy
matplotlib
pandas
