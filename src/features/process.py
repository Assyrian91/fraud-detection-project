import os

import pandas as pd

df = pd.read_csv("data/processed/clean.csv")

# Drop irrelevant columns
features = df.drop(columns=["Time", "Amount"], errors="ignore")

# Ensure target column is preserved and not duplicated
if "churn" in df.columns and "churn" not in features.columns:
    features["churn"] = df["churn"]

# Save clean features
os.makedirs("data/processed", exist_ok=True)
features.to_csv("data/processed/features.csv", index=False)
