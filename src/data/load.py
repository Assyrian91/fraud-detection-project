import pandas as pd


def load_data():
    df = pd.DataFrame(
        {"customer_id": [1, 2, 3], "churn": [0, 1, 0], "age": [25, 40, 30]}
    )
    df.to_csv("data/raw/data.csv", index=False)


if __name__ == "__main__":
    load_data()
