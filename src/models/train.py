import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train():
    df = pd.read_csv("data/processed/clean.csv")
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X = pd.get_dummies(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train()
