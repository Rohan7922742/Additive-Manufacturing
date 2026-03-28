import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/processed/labels.csv"
MODEL_PATH = "saved_models/model.pkl"
FEATURE_PATH = "saved_models/feature_columns.pkl"


def is_printable(row):
    if row["overhang_ratio"] > 0.85:
        return 0
    if row["min_thickness"] < 2:
        return 0
    return 1


def train():

    os.makedirs("saved_models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    df["printable"] = df.apply(is_printable, axis=1)

    X = df.drop(columns=["label", "file", "printable"])
    y = df["printable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("Model Accuracy:", acc)

    joblib.dump(model, MODEL_PATH)

    joblib.dump(list(X.columns), FEATURE_PATH)

    print("Model saved to:", MODEL_PATH)
    print("Feature columns saved to:", FEATURE_PATH)


if __name__ == "__main__":
    train()