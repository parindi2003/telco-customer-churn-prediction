from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Telco Customer Churn Prediction API")

# Load the saved model, scaler, and feature columns
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')


class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    PaperlessBilling: int
    MonthlyCharges: float
    TotalCharges: float
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaymentMethod: str


@app.get("/")
def read_root():
    return {"message": "Telco Customer Churn Prediction API is running"}


@app.post("/predict")
def predict_churn(customer: Customer):
    # Convert input into a DataFrame
    input_df = pd.DataFrame([customer.dict()])

    # One-hot encode the categorical columns (same as training)
    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
    input_df = pd.get_dummies(input_df, columns=multi_cat_cols)

    # Ensure input has the exact same columns as training data
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }