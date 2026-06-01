# 🧠 Breast Cancer ML Classification & Benchmarking

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange.svg)](https://scikit-learn.org/)

This project provides a robust, end-to-end Machine Learning pipeline for classifying tumors as **Malignant (M)** or **Benign (B)** using the **Breast Cancer Wisconsin Diagnostic Dataset**. It features a core benchmarking script, an interactive Streamlit Web Dashboard, and an exploratory Jupyter Notebook.

## ✨ Features

- **Model Benchmarking:** Compares Support Vector Machines (SVM) against Random Forest and Logistic Regression using `scikit-learn` Pipelines.
- **Interactive Web App:** A user-friendly Streamlit dashboard (`app.py`) for real-time hyperparameter tuning and model evaluation.
- **GridSearch Optimization:** Automated hyperparameter tuning (C, gamma, kernel type) using cross-validation.
- **Advanced Visualizations:** Includes 2D PCA decision boundaries and a Seaborn-based Confusion Matrix.
- **Interactive Notebook:** A well-documented Jupyter notebook for step-by-step learning.

## 📂 Repository Structure

| File / Component | Description |
|------------------|-------------|
| `app.py` | Interactive **Streamlit Web Dashboard** |
| `svm_breast_cancer.py` | Core automated ML pipeline and model benchmarking script |
| `svm_breast_cancer.ipynb` | Jupyter notebook for interactive step-by-step exploration |
| `breast-cancer.csv` | Cleaned input dataset |
| `confusion_matrix.png` | Visualized evaluation metrics of the best performing model |
| `svm_linear_pca.png` | PCA Decision boundary using SVM with Linear Kernel |
| `svm_rbf_pca.png` | PCA Decision boundary using SVM with RBF Kernel |

---

## 🚀 Getting Started

### 1. Installation

Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

### 2. Running the Interactive Dashboard

Launch the Streamlit web application to explore the models interactively:

```bash
streamlit run app.py
```

### 3. Running the Core Pipeline

Execute the main script to run the benchmarking, hyperparameter tuning, and generate visualizations:

```bash
python svm_breast_cancer.py
```

---

## 📊 Model Evaluation & Benchmarking

The `svm_breast_cancer.py` script automatically evaluates multiple algorithms:
*   Logistic Regression
*   Random Forest
*   Support Vector Machines (SVM)

The best SVM model is further tuned using **GridSearchCV**:
*   `C`: `[0.1, 1, 10]`
*   `gamma`: `['scale', 0.01, 0.1, 1]`

### ✅ Output Visualizations
*(Note: Visuals are automatically generated when running the pipeline script)*
![Image](https://github.com/user-attachments/assets/1ef05814-ecad-4a9c-8408-f19d7bd8e585)
![Image](https://github.com/user-attachments/assets/7c5670d6-1c67-4597-8ff1-8ec289f9fc2b)

---

## ✍️ Author
**RANGESHPANDIAN PT**
