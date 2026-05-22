# Loan Default Classification – ML Algorithms | Python

## Project Overview
A comparative machine learning study on loan portfolio health, benchmarking four classification algorithms 
to predict the probability of default on an imbalanced dataset (22% default rate, 45,000 records). 
The study evaluates model performance with a focus on minimising false negatives — missed defaults — 
which carry the highest financial risk for lenders.

## Tools & Technologies
- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Models:** Logistic Regression, Decision Tree, Random Forest, AdaBoost
- **Techniques:** One-Hot Encoding, Standard Scaling, IQR Outlier Detection, Feature Importance Analysis, Partial Dependence Plots, ROC-AUC Analysis

## Dataset
- 45,000 loan records with 13 features including borrower demographics, loan characteristics and credit history
- Target variable: `loan_status` (0 = no default, 1 = default)
- Class imbalance: 22.2% default rate

## Methodology Summary
1. **EDA & Preprocessing** — distribution analysis via histograms and count plots; correlation heatmap; IQR-based outlier detection; one-hot encoding for categorical variables (`loan_intent`, `person_education`, `person_home_ownership`); binary mapping for `previous_loan_defaults_on_file`; StandardScaler applied to all numeric features; 80/20 train-test split
2. **Model Training** — four classifiers benchmarked: Logistic Regression (max_iter=1000), Decision Tree, Random Forest, and AdaBoost (all with random_state=42 for reproducibility); predictions evaluated via confusion matrix, classification report, and accuracy score
3. **Model Evaluation & Interpretation** — ROC-AUC curve for Logistic Regression (AUC = 0.85); feature importance and partial dependence plots for Random Forest confirming `loan_int_rate` and `person_income` as top predictors; AdaBoost error-rate-per-estimator plot showing convergence stabilisation around 20–30 estimators

## Key Findings

### Model Comparison
| Model               |  Accuracy | F1 (Class 1) | Recall (Class 1) | Precision (Class 1)  | False Negatives |
|---------------------|-----------|--------------|------------------|----------------------|-----------------|
| Logistic Regression | 84.14%    | 0.57         | 0.48             | 0.72                 | 1,047           |
| Decision Tree       | 86.48%    | 0.70         | 0.72             | 0.69                 | 571             |
| AdaBoost            | 87.10%    | 0.68         | 0.62             | 0.76                 | 770             |
| **Random Forest**   | **90.41%**| **0.76**     | **0.67**         | **0.87**             | **659**         |

### Why Random Forest Won
- Highest overall accuracy (90.41%) and best F1-score for the minority class (0.76)
- Highest precision for defaults (0.87) — most reliable when flagging a default
- Cut false negatives by 37% compared to Logistic Regression (1,047 → 659)
- Lowest false positives (204) among all models — reducing unnecessary rejections

### Top Predictive Features (Random Forest Importance)
1. `loan_int_rate` — higher interest rates strongly predict default
2. `person_income` — higher income reduces default probability (non-linear)
3. `loan_percent_income` — loan burden relative to income is a key risk signal
4. `loan_amnt` — loan size contributes moderately

### Notable EDA Findings
- `loan_percent_income` (r = 0.38) and `loan_int_rate` (r = 0.33) show the 
  strongest linear correlations with default
- `person_age` and `person_emp_exp` are highly collinear (r = 0.95)
- Most common loan intent: Education, followed by Medical and Personal
- Credit scores concentrated between 600–700 across the dataset

## How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn openpyxl`
3. Run `loan_default_classification.ipynb` end-to-end in Jupyter or Google Colab

## Key Takeaway
Random Forest is the recommended model for this problem. While Decision Tree 
achieves slightly higher recall for defaults (0.72 vs 0.67), Random Forest's 
superior precision (0.87 vs 0.69) and dramatically lower false positives make it 
more suitable for real-world deployment where both missed defaults and unnecessary 
rejections carry financial cost.

## Limitations & Extensions
- **Class imbalance unaddressed** — no SMOTE or class-weight adjustment was applied; oversampling or cost-sensitive learning could meaningfully improve recall for the minority class across all models
- **Outliers retained** — IQR flagged data quality issues (e.g., age = 144, employment experience = 125 years) that were identified but not removed; cleaning these may improve model reliability
- **No hyperparameter tuning** — all models used default parameters; GridSearchCV or RandomizedSearchCV on Random Forest and AdaBoost could push performance further
- **AdaBoost convergence** — error rate stabilises around 20–30 estimators, suggesting the default of 50 is sufficient but tuning `n_estimators` and `learning_rate` is a logical next step
- **XGBoost or LightGBM** — gradient boosting frameworks not benchmarked here; likely to outperform AdaBoost on this imbalanced tabular dataset
