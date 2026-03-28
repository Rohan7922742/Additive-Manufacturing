import joblib
import pandas as pd

from src.parsing.load_mesh import load_mesh
from src.features.extract_all import extract_features


MODEL_PATH = "saved_models/model.pkl"


def predict_printability(stl_path):

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load("saved_models/feature_columns.pkl")
    mesh = load_mesh(stl_path)

    features = extract_features(mesh)

    X = pd.DataFrame([features])
    feature_columns = joblib.load("saved_models/feature_columns.pkl")

    prediction = model.predict(X)[0]

    probability = model.predict_proba(X)

    return prediction, probability


if __name__ == "__main__":

    stl_file = "samples/test_model.stl"

    pred, prob = predict_printability(stl_file)

    print("Prediction:", pred)
    print("Probability:", prob)