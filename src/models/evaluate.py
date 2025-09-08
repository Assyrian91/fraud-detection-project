import json
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate():
    df = pd.read_csv("data/processed/clean.csv")
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X = pd.get_dummies(X)

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds, output_dict=True)

    with open("metrics.json", "w") as f:
        json.dump({"accuracy": acc, "report": report}, f, indent=2)


if __name__ == "__main__":
    evaluate()
