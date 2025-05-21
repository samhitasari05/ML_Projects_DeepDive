# 🎯 Dimensionality Decoded: PCA for Model Optimization

This project explores the use of **Principal Component Analysis (PCA)** to reduce dimensionality and improve model interpretability. It also compares the performance of multiple classifiers after dimensionality reduction.

🧪 This was completed as part of AIT 636 (Machine Learning) at George Mason University and demonstrates how PCA can simplify data while preserving the most important variance for classification.

---

## 🧠 What is PCA?

**Principal Component Analysis (PCA)** is a technique used to reduce the number of input variables in a dataset while retaining as much information (variance) as possible.

- It transforms data into a new coordinate system
- Keeps only the most important "principal components"
- Helps with visualization, speed, and reducing noise

---

## 🚀 What This Project Covers

- Standardizing data for PCA
- Performing PCA transformation using `scikit-learn`
- Reducing feature space to 2 components
- Training and evaluating multiple classifiers on PCA-transformed data:
  - SVM (Linear, Poly, RBF, Sigmoid)
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Naive Bayes
  - K-Nearest Neighbors
  - Ridge Regression
  - Lasso Regression
  - Ordinary Least Squares

---

## 📊 Results Summary

> ✅ PCA successfully reduced dimensionality to 2 components  
> ✅ All classifiers were trained on PCA-reduced features  
> ✅ Model performance was visualized via classification region plots  
> ✅ PCA helped improve interpretability and visualization of decision boundaries

---


