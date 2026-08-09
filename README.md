

# 🎓 Student Exam Score Prediction: Linear Models vs. ANN — A Controlled Comparison
 
This repository contains an end-to-end machine learning pipeline to predict student exam performance. Beyond fitting models, the project is built around a specific question: **does a neural network actually add predictive value over classical linear models here, or does it only look like it does?** The pipeline covers leakage-free preprocessing, OLS assumption validation, interaction-based feature engineering, five distinct ANN architectures, and 5-fold cross-validation — with every experimental dead end documented, not just the final result.
 
**Dataset:** [Exam Score Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/kundanbedmutha/exam-score-prediction-dataset)
 
---
 
## 📊 Performance Report
 
### Phase 1: Baseline Models (Raw Features)
 
Linear models outperformed tree-based models from the start, an early signal that the relationship between core features (study hours, attendance, sleep) and exam score is largely linear/additive rather than interaction-driven.
 
| Algorithm | R² | RMSE |
| :--- | :--- | :--- |
| **Lasso Regression (L1)** | **0.7332** | **9.7715** |
| **Ridge Regression (L2)** | 0.7330 | 9.7725 |
| **Linear Regression** | 0.7330 | 9.7725 |
| **XGBoost** | 0.7066 | 10.2441 |
 
### Phase 2: Post-Feature Engineering
 
Three interaction terms were introduced — **Study Efficiency** (study_hours × sleep_hours), **Total Engagement** (study_hours × class_attendance), and **Rest Quality Index** (sleep_hours × sleep_score) — with their raw parent columns dropped.
 
| Algorithm | R² | RMSE |
| :--- | :--- | :--- |
| **XGBoost (Post-Eng)** | **0.7133** | **10.1268** |
| **Lasso Regression** | 0.7082 | 10.2161 |
| **Ridge Regression** | 0.7080 | 10.2207 |
| **Baseline LR** | 0.7079 | 10.2208 |
 
> **Important correction from Phase 3's diagnostics:** every model's R² *dropped* after feature engineering (Lasso 0.7332→0.7082, Ridge 0.7330→0.7080). This isn't feature engineering "helping some models more than others" — it's a straightforward loss of information. Replacing raw variables with their products, and dropping the originals, removes information a linear model needs to represent an additive relationship. XGBoost's rise from 0.7066→0.7133 is real, but it's still ~0.02 R² *below* what plain linear regression achieves on the untouched raw features (0.7330). Phase 3 traces this precisely.
 
### Phase 3: ANN Experimentation
 
Five architectures were tested, each built to test a specific hypothesis raised by the previous result — not just increasing complexity for its own sake.
 
| # | Architecture | Feature Set | R² | Why it was tried / what it showed |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Plain MLP (dropout 0.2, L2) | Engineered | 0.7073 | Control test — underperformed Ridge; train_loss > val_loss signaled over-regularization |
| 2 | Plain MLP (dropout 0.1, no L2) | Engineered | 0.7144 | Loosened regularization closed the train/val gap, but a real gap to Ridge remained |
| 3 | Wide & Deep (joint training, L2 wide branch) | Engineered | 0.7135 | Tested whether an explicit linear branch + deep branch could match Ridge — no improvement |
| 4 | Wide & Deep (joint, no L2, tuned LR) | Engineered | 0.7150 | Ruled out L2 strength as the cause; pointed to gradient-imbalance in joint training (deep branch's ~3,585 params dominating the wide branch's ~23) |
| 5 | **Staged Residual Wide+Deep** (Ridge fit first, MLP trained only on residuals) | Engineered | **0.7192** | Fixed the imbalance by decoupling training entirely — genuine improvement over Ridge on this feature set |
| 5b | Staged Residual Wide+Deep (same architecture) | **Raw** | 0.7329 | Matches Ridge on raw features almost exactly (0.7330) — confirms Phase 3 #5's gain was recovered *lost linear signal*, not new non-linear structure |
 
**The key finding:** the same staged architecture that appeared to beat Ridge on engineered features (0.7080→0.7192) found *nothing* when given the raw features directly (0.7330→0.7329, a difference of -0.0001). The residual model's validation loss diverged from epoch 1 on raw features — a clear signature of fitting pure noise. This confirms the Phase 3 #5 "improvement" was the network partially reconstructing linear information the feature engineering had deleted, not evidence of real non-linearity in the data.
 
### Phase 4: 5-Fold Cross-Validation (Raw Features)
 
A single train/test split can make a close result look more or less conclusive than it is. 5-fold CV was run to confirm the raw-feature finding holds statistically.
 
| Model | R² (mean ± std) | RMSE (mean ± std) |
| :--- | :--- | :--- |
| Ridge | 0.7315 ± 0.0046 | 9.80 ± 0.05 |
| **Staged Wide+Deep** | **0.7315 ± 0.0046** | **9.80 ± 0.06** |
| XGBoost | 0.7025 ± 0.0051 | 10.31 ± 0.06 |
 
Ridge and Staged Wide+Deep are statistically indistinguishable — matching to four decimal places on the mean, and tracking together fold-by-fold (not just on average). XGBoost underperforms consistently by ~2.9 R² points across every fold.
 
---
 
## 🔍 Key Insights & Impact Analysis
 
### 1. The relationship is genuinely linear — and the ANN experiments prove it, not just assume it
 
Five ANN architectures were tested specifically to falsify the "this is a linear problem" hypothesis. All five failed to beat Ridge when given equivalent information (the raw feature set), and the one architecture that appeared to succeed was traced to a feature-engineering artifact rather than a modeling win. Cross-validation confirms this isn't a single-split coincidence.
 
### 2. XGBoost consistently underperforms linear models — corroborating, not contradicting, that conclusion
 
Tree ensembles earn their advantage over linear models specifically when there's non-linear interaction structure to exploit. XGBoost's consistent underperformance (Phase 1, Phase 2, and Phase 4) independently supports the same conclusion the ANN experiments reached: this dataset doesn't have meaningful feature-interaction structure for a flexible model to find.
 
### 3. Feature engineering can *destroy* signal, not just add it
 
The multiplicative interaction terms in Phase 2 replaced raw continuous variables their linear models needed directly. This is a general lesson worth internalizing: adding interaction terms *alongside* raw features preserves both individual and joint effects; replacing raw features *with* interaction terms can silently discard information, even when the interaction terms themselves are conceptually reasonable.
 
### 4. Regularization diagnostics matter as much as architecture choice
 
The first ANN attempt underperformed not because the architecture was wrong, but because dropout(0.2) + L2 were too aggressive for a low-complexity target function — visible directly in a train_loss > val_loss inversion in the training curve. Diagnosing this before concluding "ANN doesn't work" avoided a false negative.
 
### 5. Feature Importance & Behavioral Trends (from OLS/Ridge coefficients)
 
* **High impact:** `study_hours` and `class_attendance` are the primary drivers of exam score.
* **Study method:** `self-study` and `online videos` show negative coefficients relative to the baseline category, suggesting these students may benefit more from structured or collaborative study environments.
---
 
## 🛠️ Methodology & Best Practices
 
* **Data Leakage Prevention:** All transformations (scaling, one-hot encoding) were implemented using **Scikit-Learn Pipelines**, and refit independently within each cross-validation fold — the test/validation fold never influences preprocessing statistics.
* **OLS Assumption Validation:** Before treating any linear R² as a trustworthy baseline, VIF (multicollinearity), residual homoscedasticity, Q-Q normality, and Durbin-Watson (autocorrelation, ≈2.02 — no autocorrelation) were checked and confirmed sound.
* **Multicollinearity Management:** VIF analysis confirmed no severe collinearity in either feature set (all values well under 5), ruling out multicollinearity as a confound in the model comparisons.
* **Categorical Handling:** Used `drop='first'` in One-Hot Encoding to avoid the Dummy Variable Trap and keep the linear models' coefficients stable and interpretable.
* **Controlled ANN Experimentation:** Every architecture change was made to test a specific hypothesis raised by the previous result (regularization → architecture → training dynamics → feature representation), not as an unstructured hyperparameter search.
* **Statistical Validation:** 5-fold cross-validation was used to confirm the final result wasn't an artifact of one particular train/test split.
---
 
## 🚀 Future Work
 
1. **Fix the feature engineering, properly this time:** add interaction terms *alongside* raw features rather than replacing them, and check whether this closes any remaining gap without needing the ANN at all.
2. **SHAP analysis:** run SHAP on the raw-feature Ridge model and cross-check against its OLS coefficients for a fully triangulated interpretability story.
3. **Stacking ensemble:** use out-of-fold predictions from Ridge and XGBoost as inputs to a meta-regressor, evaluated with the same leakage-safe CV protocol used in Phase 4.
4. **CatBoost:** test native categorical handling (e.g. for `course`) against one-hot encoding.
5. **Data expansion:** the ~0.73 ceiling is likely a data limitation, not a modeling one — additional feature types (e.g. prior academic performance, socioeconomic indicators) are a more promising path to meaningfully improving R² than further model complexity.
---
 
## 📁 Repository Structure
 
```
├── data/
│   └── Exam_Score_Prediction.csv
├── notebooks/
│   ├── Exam_Score_Prediction.ipynb
└── README.md
```
 

















