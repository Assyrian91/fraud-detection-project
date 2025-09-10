import os

import pandas as pd

df = pd.read_csv("data/raw/data.csv")

df.dropna(inplace=True)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/clean.csv", index=False)
