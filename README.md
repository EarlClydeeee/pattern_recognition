# Student Performance Analysis

A machine learning project that explores the academic and socio-economic factors influencing student exam scores. The analysis covers the full pipeline from data preparation through both supervised (regression) and unsupervised (clustering) learning.

---

## Dataset

**Source:** [Kaggle — Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) (`lainguyn123/student-performance-factors`)

- **Records:** 6,607 students (synthetically generated)
- **Features:** 20 academic and socio-economic attributes
- **Target Variable:** `Exam_Score` (numerical)

Key features include study habits (`Hours_Studied`, `Tutoring_Sessions`), environmental factors (`Internet_Access`, `Family_Income`, `Teacher_Quality`), and personal metrics (`Motivation_Level`, `Physical_Activity`).

---

## Project Structure

```
student_performance.ipynb   # Main analysis notebook
README.md
```

The notebook is organized into three parts:

### Part A — Data Preparation
- Load dataset via `kagglehub`
- Exploratory data analysis (shape, dtypes, value counts)
- Handle missing values (mode imputation for `Teacher_Quality`, `Parental_Education_Level`, `Distance_from_Home`)
- Detect and remove duplicates
- Outlier removal using selective IQR (1.5× for most features, 2.0× for `Exam_Score`)
- One-Hot Encoding of categorical variables
- Feature scaling with `StandardScaler`

### Part B — Supervised Learning (Regression)
Train and evaluate five regression models to predict `Exam_Score`:

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularization (α = 0.001) |
| Lasso Regression | L1 regularization (α = 0.001) |
| ElasticNet | Combined L1 + L2 (α = 0.001) |
| Random Forest Regressor | Ensemble tree model |

Evaluation metrics: MAE, MSE, R² (test and train). Results are visualized via actual vs. predicted scatter plots for each model.

### Part C — Unsupervised Learning (K-Means Clustering)
- Elbow method to determine optimal *k*
- Automatic *k* selection via second-order curvature of inertia
- PCA (2 components) for cluster visualization
- Master cluster profiles (mean numerical + modal categorical features)
- Z-score heatmap highlighting key differentiators across clusters
- Per-cluster signature report (top 2 peaks and valleys by Z-score)

---

## Requirements

```bash
pip install kagglehub pandas numpy matplotlib seaborn scikit-learn
```

A Kaggle account and API token are required to download the dataset via `kagglehub`.

---

## Usage

Open and run the notebook top-to-bottom in a Jupyter environment:

```bash
jupyter notebook student_performance.ipynb
```

All three parts are self-contained and run sequentially.