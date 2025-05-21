# 🎯 Targeting Risk: A Beginner's Journey with Logistic Regression

Welcome to a project that demystifies one of the most foundational tools in data science — **Logistic Regression**. This project walks through how we can use it to make smart, data-backed decisions when it comes to binary outcomes — like predicting health risks.

🧪 Completed as part of my graduate coursework (AIT 636 – Machine Learning at George Mason University), this project was a hands-on application of logistic regression using real-world style data.

---

## ❓ What Is Logistic Regression?

> Think of logistic regression as a **yes or no predictor**.

You give it a set of inputs (like age, glucose level, BMI), and it gives you a probability that something will happen — like whether a person might have a medical condition (`1`) or not (`0`).

Instead of drawing a straight line (like linear regression), **logistic regression fits an S-shaped curve** that maps input values to a probability between 0 and 1.

It’s widely used for:
- Diagnosing diseases
- Email spam filtering
- Customer churn prediction
- Loan approval predictions

---

## 🧠 What This Project Covers

- **Data preprocessing**: Cleaning and preparing structured health data
- **Feature selection**: Choosing which variables affect the target
- **Logistic Regression modeling** using `scikit-learn`
- **Performance metrics**: Accuracy, precision, recall, F1-score
- **Interpretation**: Understanding the model’s predictions and what influences them

---

## 🧬 Dataset Overview

The dataset contains health-related measurements like:
- Age
- Glucose levels
- Blood pressure
- BMI
- Insulin levels

And the goal is to predict:
> Will this person be diagnosed with the condition?  
> `1 = Yes`, `0 = No`

---

## 📈 Results Summary

The logistic regression model was trained and tested using multiple configurations, and its performance was evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

Model results were summarized, interpreted, and visualized in both the notebook and accompanying PDF report.

---

## 🗂️ Project Files

- `Assignment6B_AIT636_001_Samhita_Sari.ipynb` – Core implementation in Jupyter Notebook
- `Assignment6B_AT636_001_SamhitaSarikonda.pdf` – Report with evaluation and explanations
- `README.md` – This file
- `requirements.txt` – Python libraries used

---

## ⚙️ Libraries Used

```bash
scikit-learn
numpy
pandas
matplotlib

