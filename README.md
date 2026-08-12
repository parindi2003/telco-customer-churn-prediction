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