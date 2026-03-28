import streamlit as st
import tempfile
from PIL import Image
import os
import joblib
import pandas as pd

import sys

import os

# --- Set project root so Python can find 'src' ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # app/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   # AMML/
sys.path.append(PROJECT_ROOT)

# Now imports from src will work
from src.parsing.load_mesh import load_mesh
from src.features.extract_all import extract_features

import streamlit as st
import tempfile
import pyvista as pv
from PIL import Image
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       
PROJECT_ROOT = os.path.dirname(BASE_DIR)                   
sys.path.append(PROJECT_ROOT)

from src.parsing.load_mesh import load_mesh

from src.features.extract_all import extract_features


MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "model.pkl")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

st.title("3D Printability Predictor")
st.write("Upload an STL file to check if it is printable.")

uploaded_file = st.file_uploader("Upload STL File", type=["stl"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    st.success("File uploaded successfully!")

    try:
        mesh = load_mesh(temp_path)
        features = extract_features(mesh)

        X = pd.DataFrame([features])[feature_columns]

        probs = model.predict_proba(X)[0]  
        pred_index = probs.argmax()
        prediction = model.classes_[pred_index]

        st.subheader("Prediction Result")
        st.write("Prediction:", prediction)

        st.subheader("Prediction Probabilities")
        st.write(f"Printable: {probs[0]:.2f}")
        st.write(f"Not Printable: {probs[1]:.2f}")

    except Exception as e:
        st.error(f"Failed to process STL file: {e}")
