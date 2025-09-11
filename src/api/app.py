import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("models/model.pkl")

app = FastAPI()


class Features(BaseModel):
    tenure: int
    monthly_charges: float
    contract_type: str


class InputData(BaseModel):
    customer_id: str
    features: Features


@app.post("/predict")
def predict(data: InputData):
    features = {
        "tenure": data.features.tenure,
        "monthly_charges": data.features.monthly_charges,
        "contract_type": data.features.contract_type,
    }
    df = pd.DataFrame([features])
    df = pd.get_dummies(df)

    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]

    prediction = model.predict(df)[0]
    confidence = (
        model.predict_proba(df)[0][prediction]
        if hasattr(model, "predict_proba")
        else None
    )

    return {
        "customer_id": data.customer_id,
        "churn": bool(prediction),
        "confidence": round(float(confidence), 4) if confidence else "N/A",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
