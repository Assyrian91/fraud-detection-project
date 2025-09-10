import os

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_params(params_path: str = "params.yaml") -> dict:
    """Loads parameters from a YAML file."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_and_split_data(
    features_path: str,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple:
    """Loads data, splits into features and target, and creates train/test sets."""
    df = pd.read_csv(features_path)

    if not target_col in df.columns:
        msg = f"Target column '{target_col}' not found in features file."
        raise ValueError(msg)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, X.columns.to_list()


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    penalty: str,
    C: float,
    max_iter: int,
) -> LogisticRegression:
    """Initializes and trains a Logistic Regression model."""
    clf = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def save_artifacts(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    columns: list,
    model_path: str,
    test_out_path: str,
) -> None:
    """Saves the trained model and the test dataset."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
    np.savez_compressed(
        test_out_path,
        X_test=X_test.to_numpy(),
        y_test=y_test.to_numpy(),
        columns=columns,
    )


def main() -> None:
    """Main function to run the training pipeline."""
    params = load_params()

    features_path = params["data"]["features"]
    test_out_path = params["data"]["test_split"]
    model_path = params["train"]["model_path"]
    target_col = params["train"]["target"]
    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]
    penalty = params["train"]["penalty"]
    C = params["train"]["C"]
    max_iter = params["train"]["max_iter"]

    X_train, X_test, y_train, y_test, columns = load_and_split_data(
        features_path,
        target_col,
        test_size,
        random_state,
    )

    clf = train_model(X_train, y_train, penalty, C, max_iter)

    save_artifacts(
        clf,
        X_test,
        y_test,
        columns,
        model_path,
        test_out_path,
    )

    print(f"Saved model to {model_path}")
    print(f"Saved test split to {test_out_path}")


if __name__ == "__main__":
    main()
