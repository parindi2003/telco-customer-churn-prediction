# Telco Customer Churn Prediction

Predicting which customers are likely to leave a telecom company, using
customer demographics, subscribed services, and billing information.

## Dataset
- Source: Telco Customer Churn (Kaggle)
- 7,043 customers, 21 features
- Target: Churn (Yes/No)

## Key EDA Insights

### Class Imbalance
Churn distribution: No = 73.5%, Yes = 26.5%. This is an imbalanced 
dataset, meaning accuracy alone is misleading — a model that always 
predicts "No" would score 73.5% accuracy without learning anything. 
Precision, Recall, and F1-score are used instead, with priority on 
Recall since missing an actual churner has higher business cost than 
a false alarm.

### Data Quality Fix
TotalCharges was stored as text with 11 hidden empty values, all 
belonging to customers with tenure = 0 (brand new customers). Fixed by 
converting to numeric and filling missing values with 0.

### Contract Type Strongly Predicts Churn
- Month-to-month: ~43% churn rate
- One-year: ~11% churn rate
- Two-year: ~3% churn rate

Business recommendation: incentivize longer-term contracts to reduce churn.

## How to Run
(will add after project is complete)

## Tech Stack
Python, pandas, scikit-learn, FastAPI

## Feature Engineering

Before training any models, the raw data needed to be converted into a 
format machine learning algorithms can understand — mostly numbers, not text.

**Steps taken:**

1. **Removed customerID** — this is just a unique identifier for each 
   customer and carries no useful information for predicting churn.

2. **Converted Yes/No columns to 1/0** — columns like `Partner`, 
   `Dependents`, `PhoneService`, and the target column `Churn` itself 
   only had two possible values, so they were mapped directly to 1 
   (Yes) and 0 (No).

3. **One-Hot Encoded multi-category columns** — columns like `Contract`, 
   `InternetService`, and `PaymentMethod` have three or more categories 
   with no natural order between them. Instead of assigning arbitrary 
   numbers (which would falsely imply ranking), each category was split 
   into its own True/False column. This expanded the dataset from 20 to 
   31 columns.

4. **Split into training and test sets** — 80% of the data was used for 
   training the models, and 20% was held back to test how well the 
   models perform on data they haven't seen before. The split was 
   stratified, meaning both sets preserve the same 73.5% / 26.5% 
   churn ratio as the original data — important for keeping evaluation 
   fair on this imbalanced dataset.

   ## Model Training & Evaluation

With the data cleaned and encoded, three different classification models 
were trained and compared to find the best approach for predicting churn.

### Why compare multiple models?

Different algorithms make different assumptions about the data. Logistic 
Regression assumes a roughly linear relationship between features and 
the outcome, while Random Forest and XGBoost are tree-based ensemble 
methods that can capture more complex, non-linear patterns. Training all 
three and comparing their results is standard practice — there's no way 
to know in advance which will perform best on a given dataset.

### Model 1: Logistic Regression (Baseline)

Used as a simple, interpretable baseline. Since Logistic Regression is 
sensitive to the scale of input features, the data was standardized 
using `StandardScaler` before training — this improved every metric 
slightly compared to using the raw, unscaled features.

| Metric | Score |
|--------|-------|
| Accuracy | 80.7% |
| Precision | 65.8% |
| Recall | 56.7% |
| F1-Score | 60.9% |

### Model 2: Random Forest

An ensemble of 100 decision trees, trained with default hyperparameters. 
Tree-based models don't require feature scaling, so the raw features 
were used directly.

| Metric | Score |
|--------|-------|
| Accuracy | 78.4% |
| Precision | 62.1% |
| Recall | 48.1% |
| F1-Score | 54.2% |

### Model 3: XGBoost

A gradient boosting model, also trained with default hyperparameters 
(100 estimators).

| Metric | Score |
|--------|-------|
| Accuracy | 77.6% |
| Precision | 58.9% |
| Recall | 52.1% |
| F1-Score | 55.3% |

### Model Comparison & Selection

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | **80.7%** | **65.8%** | **56.7%** | **60.9%** |
| Random Forest | 78.4% | 62.1% | 48.1% | 54.2% |
| XGBoost | 77.6% | 58.9% | 52.1% | 55.3% |

**Logistic Regression was selected as the best-performing model**, 
outperforming both ensemble methods on every metric. This was somewhat 
unexpected, since tree-based ensembles often outperform simpler linear 
models. Two likely explanations:

1. Random Forest and XGBoost were trained with default hyperparameters, 
   without tuning (e.g., via GridSearchCV) — a fairer comparison would 
   require tuning all three models before drawing final conclusions.
2. The relationship between the features and churn in this dataset 
   appears to be largely linear/additive, which naturally favors a 
   linear model like Logistic Regression.

This is a useful reminder that more complex models don't automatically 
perform better — model selection should always be driven by actual 
evaluation on the data, not by assumptions about model complexity.

### Confusion Matrix (Logistic Regression)

|                    | Predicted: No Churn | Predicted: Churn |
|--------------------|----------------------|-------------------|
| **Actual: No Churn** | 925 (True Negative)  | 110 (False Positive) |
| **Actual: Churn**    | 162 (False Negative) | 212 (True Positive)  |

The model correctly identifies 212 out of 374 actual churners (56.7% 
recall). The 162 missed churners (false negatives) are the most costly 
errors from a business perspective — these are customers who leave 
without the company having any warning to intervene. Reducing this 
number would be the top priority for future model improvement, since 
a missed churner has a higher business cost than a false alarm (an 
unnecessary retention offer sent to a loyal customer).

### Feature Importance (Logistic Regression Coefficients)

| Feature | Coefficient | Effect |
|---------|-------------|--------|
| tenure | -1.24 | Strongest predictor — longer-tenured customers are far less likely to churn |
| MonthlyCharges | -0.92 | Counter-intuitive at first glance; likely reflects a correlation with tenure rather than a direct causal effect |
| InternetService: Fiber optic | +0.78 | Fiber optic customers show higher churn risk |
| Contract: Two year | -0.59 | Confirms the earlier EDA finding — long-term contracts reduce churn |
| TotalCharges | +0.51 | Higher lifetime spend is associated with higher churn risk |

**Business takeaway:** Customer tenure and contract length are the 
strongest predictors of churn. Retention efforts are likely to have the 
biggest impact if focused on the critical early period of a customer's 
lifecycle, and on converting month-to-month customers to longer-term 
contracts.