import os

import pandas as pd

df = pd.read_csv("data/processed/clean.csv")

features = df.drop(columns=["Time", "Amount"], errors="ignore")

# Preserve target column
features["churn"] = df["churn"]

os.makedirs("data/processed", exist_ok=True)
features.to_csv("data/processed/features.csv", index=False)
