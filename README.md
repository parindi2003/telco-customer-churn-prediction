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