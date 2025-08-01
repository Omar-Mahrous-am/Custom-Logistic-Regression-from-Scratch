# Logistic Regression from Scratch

This project implements **Logistic Regression** from scratch using only NumPy. It includes the full training loop with gradient descent, a sigmoid function, thresholded prediction, and accuracy evaluation using a synthetic binary classification dataset.

---

## 📌 Features

- ✅ Pure Python (only NumPy used for model logic)
- ✅ Binary classification using sigmoid activation
- ✅ Manual gradient descent implementation
- ✅ Threshold customization for prediction
- ✅ Accuracy evaluation with scikit-learn

---

## 🧠 Algorithm Overview

Logistic Regression is a linear model used for binary classification. It predicts the probability that an input belongs to a certain class using the **sigmoid function**. This implementation includes:

- **Sigmoid activation** for probabilities  
- **Binary cross-entropy loss (implicitly minimized)**
- **Gradient descent** to optimize weights
- **Prediction threshold** for classification (default: 0.5)

---

## 🧪 Dataset

We use a **synthetic dataset** generated with `sklearn.datasets.make_classification()` for demonstration. It has:

- 500 samples
- 5 features
- 2 target classes

---

## 🛠️ Installation

You only need NumPy and scikit-learn:

```bash
pip install numpy scikit-learn
