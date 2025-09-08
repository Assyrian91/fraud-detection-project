import pandas as pd


def process_data():
    df = pd.read_csv("data/raw/data.csv")
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 50, 100], labels=["young", "middle", "senior"]
    )
    df.to_csv("data/processed/clean.csv", index=False)


if __name__ == "__main__":
    process_data()
