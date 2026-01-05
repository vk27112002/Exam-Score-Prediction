# 🎓 Student Exam Score Prediction: Comparative Analysis & Feature Engineering

This repository contains an end-to-end machine learning pipeline to predict student exam performance. The project focuses on handling **multicollinearity**, implementing **leakage-free preprocessing**, and evaluating the impact of **interaction-based feature engineering** across multiple algorithms.

## 📊 Performance Report

### Phase 1: Baseline Models (Raw Features)
In the initial phase, linear models outperformed tree-based models, suggesting a strong linear relationship between the core features (like study hours) and the target score.

| Algorithm | R-squared ($R^2$) | RMSE |
| :--- | :--- | :--- |
| **Lasso Regression (L1)** | **0.7332** | **9.7715** |
| **Ridge Regression (L2)** | 0.7330 | 9.7725 |
| **Linear Regression** | 0.7330 | 9.7725 |
| **XGBoost** | 0.7066 | 10.2441 |

### Phase 2: Post-Feature Engineering
We introduced three interaction terms: **Study Efficiency**, **Total Engagement**, and **Rest Quality Index**. To prevent multicollinearity, raw parent variables were dropped.

| Algorithm | R-squared ($R^2$) | RMSE |
| :--- | :--- | :--- |
| **XGBoost (Post-Eng)** | **0.7133** | **10.1268** |
| **Lasso Regression** | 0.7082 | 10.2161 |
| **Ridge Regression** | 0.7080 | 10.2207 |
| **Baseline LR** | 0.7079 | 10.2208 |

---

## 🔍 Key Insights & Impact Analysis

### 1. The Regularization Advantage
Initially, **Lasso (L1) Regression** was the top performer ($R^2: 0.7332$). This indicates that while many factors affect grades, some variables were likely redundant or added noise. Lasso successfully "zeroed out" the noise, providing a more generalized and accurate prediction than standard OLS.

### 2. Why XGBoost Improved After Feature Engineering
A significant observation is the rise of XGBoost in Phase 2. While linear models saw a slight drop in $R^2$ after dropping raw features, XGBoost improved from **0.7066 to 0.7133**.
* **The Reason:** Linear models struggle when you replace "raw" continuous data with products (Interactions). 
* **The Takeaway:** XGBoost successfully exploited the structured complexity of the new features, proving that tree-based models are better at capturing "synergy" between variables.



### 3. Feature Importance & Behavioral Trends
* **High Impact:** `study_hours` and `class_attendance` are the primary drivers of success.
* **Study Method:** The model revealed that `self-study` and `online videos` often had negative coefficients compared to the baseline. This suggests that students in this demographic benefit more from structured or collaborative environments.

---

## 🛠️ Methodology & Best Practices

* **Data Leakage Prevention:** All transformations (Scaling, One-Hot Encoding) were implemented using **Scikit-Learn Pipelines**. This ensures that the test set remains entirely "unseen" during the training phase.
* **Multicollinearity Management:** Conducted **Variance Inflation Factor (VIF)** analysis to ensure that the inclusion of interaction terms did not destabilize the model coefficients.
* **Categorical Handling:** Utilized `drop='first'` in One-Hot Encoding to avoid the Dummy Variable Trap, ensuring the stability of the regression models.



---

## 🚀 Future Work: Breaking the 0.75 Barrier
1. **Stacking Ensemble:** Use the predictions of Lasso and XGBoost as inputs for a Final Meta-Regressor.
2. **CatBoost:** Experiment with CatBoost to see if native handling of categorical variables (like `course`) yields better results than One-Hot Encoding.
3. **Polynomial Features:** Instead of manual interaction, use systematic polynomial expansion to find hidden non-linearities.

---

## 📁 Repository Structure
* `Exam_Score_pred.ipynb`: Full analysis and model training pipeline.
* `Exam_Score_Prediction.csv`: The raw dataset (20,000 entries).
* `requirements.txt`: List of dependencies (pandas, scikit-learn, xgboost, seaborn).

## 💻 How to Run
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/Repo-Name.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Open the Jupyter Notebook to view the full analysis.
