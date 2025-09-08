import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

with open("params.yaml") as f:
    params = yaml.safe_load(f)

features_path = params["data"]["features"]
test_out_path = params["data"]["test_split"]
model_path = params["train"]["model_path"]
target_col = params["train"]["target"]
test_size = params["train"]["test_size"]
random_state = params["train"]["random_state"]

penalty = params["train"]["penalty"]
C = params["train"]["C"]
max_iter = params["train"]["max_iter"]

df = pd.read_csv(features_path)
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in features file.")

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

clf = LogisticRegression(
    penalty=penalty,
    C=C,
    max_iter=max_iter,
    solver="lbfgs",
    n_jobs=-1
)
clf.fit(X_train, y_train)

os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(clf, model_path)

os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
np.savez_compressed(
    test_out_path,
    X_test=X_test.to_numpy(),
    y_test=y_test.to_numpy(),
    columns=X.columns.to_list()
)
print(f"Saved model to {model_path} and test split to {test_out_path}")
