# 🎯 Predicting by Proximity: A KNN Classification Walkthrough

This project explores **K-Nearest Neighbors (KNN)**, a simple yet effective classification algorithm that bases predictions on the closest data points in feature space.

🧪 This project was developed as part of AIT 636 (Machine Learning) at George Mason University. It demonstrates how local neighborhood structures can be used to classify data with high interpretability and minimal assumptions.

---

## 🧠 What is KNN?

KNN is a **lazy learning algorithm** that makes predictions by measuring the distance between a new data point and its `K` closest neighbors in the training set. It’s useful for:
- Classification
- Recommendation systems
- Pattern recognition
- Outlier detection

KNN does not require a training phase — it simply stores data and calculates distances during prediction time.

---

## 🚀 What This Project Covers

- Data normalization for distance-based algorithms
- Implementing KNN using `scikit-learn`
- Varying the number of neighbors (`K`) to find the optimal value
- Evaluating classification performance using:
  - Accuracy
  - Confusion Matrix
  - Precision / Recall / F1 Score
- Visualizing classification results

---

## 📊 Results Summary

> ✅ Found that `K = 5` yielded optimal balance between bias and variance  
> ✅ Achieved strong classification accuracy and well-distributed confusion matrix  
> ✅ Explored how different values of `K` influence model stability and generalization

*(Optional: Add a confusion matrix image or K vs Accuracy plot in `images/`)*

---

## 🗂️ Files Included

- `KNN.py` — Core KNN implementation  
- `KNN.ipynb` — Jupyter notebook version  
- `KNN_Results.pdf` — Full write-up with metrics and analysis

---

## 📚 Technologies & Libraries

```bash
scikit-learn
numpy
pandas
matplotlib

To install dependencies:
pip install -r requirements.txt
