import pandas as pd

df = pd.read_csv("data/processed/labels.csv")
print(df.columns)
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df["label"].value_counts())