# 🧠 Cracking the Kernel: A Journey into SVM Model Comparison

Welcome to a hands-on exploration of one of the most fascinating algorithms in machine learning — the **Support Vector Machine (SVM)**. In this project, I dive deep into the power of kernels and hyperparameters to answer a simple yet impactful question:

> Can the right combination of kernel and tuning turn a good model into a great one?

This project was completed as part of my graduate Machine Learning coursework at George Mason University (AIT 636), and I turned it into an insightful experiment to learn what *really* affects model performance.

---

## 🧪 The Mission: Predicting Diabetes with Precision

We use the **Pima Indians Diabetes dataset**, a well-known benchmark in healthcare analytics, to predict whether a patient is diabetic or not based on features like:
- Glucose levels
- Blood pressure
- BMI
- Insulin
- Age
- and more...

It's real-world. It's structured. And it's perfect for experimenting with SVMs.

---

## 🧩 The Experiment Setup

Imagine you're handed this dataset and a toolbox full of SVM kernels:
- 🧮 **Linear** – simple, efficient, and interpretable  
- 🔁 **Polynomial** – adding layers of flexibility with degrees  
- 🌀 **RBF (Radial Basis Function)** – the master of non-linear boundaries  
- ☯️ **Sigmoid** – the outsider, similar to neural networks

Then you begin tweaking the dial: changing **C values** (0.5, 1.0, 1.5, 2.0) to adjust regularization strength.

---

## 📈 The Findings

Over multiple runs and metric evaluations, here’s a snapshot of the F1-scores:

| Kernel    | C Value | F1 Score |
|-----------|---------|----------|
| Linear    | 1.0     | **0.63**  |
| RBF       | 1.0     | 0.61     |
| Poly (d=3)| 2.0     | 0.54     |
| Sigmoid   | 1.0     | 0.57     |

🔎 The linear kernel surprised me by outperforming others on this dataset — proof that **simplicity wins** when the data supports it.

---

## 📂 Project Files

- `AIT_636_assignment7b_svm.py` — Core implementation using `scikit-learn`
- `AIT_636_Assignment7B_SVM_SamhitaSari.pdf` — Report with visualizations and full evaluation
- `README.md` — You’re reading it 🙂
- `requirements.txt` — Library list for reproducibility

---

## ⚙️ Stack & Tools

Built with:
- Python 🐍
- scikit-learn
- NumPy
- Matplotlib (optional, for visualizations)

To install requirements:
```bash
pip install -r requirements.txt
