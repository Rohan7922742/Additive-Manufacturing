import tkinter as tk
from tkinter import filedialog
import joblib
import pandas as pd

from src.parsing.load_mesh import load_mesh
from src.features.extract_all import extract_features

MODEL_PATH = "saved_models/model.pkl"
FEATURE_PATH = "saved_models/feature_columns.pkl"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("STL Files", "*.stl")])

    if file_path:
        result_label.config(text="Processing model...")

        try:
            mesh = load_mesh(file_path)

            features = extract_features(mesh)

            X = pd.DataFrame([features])
            X = X[feature_columns]

            probability = model.predict_proba(X)[0]

            printable_percent = probability[1] * 100
            not_printable_percent = probability[0] * 100

            result_label.config(
                text=f"Printable: {printable_percent:.2f}%\nNot Printable: {not_printable_percent:.2f}%"
            )

        except Exception as e:
            result_label.config(text=f"Error: {e}")


root = tk.Tk()
root.title("3D Printability Checker")
root.geometry("400x250")

title = tk.Label(root, text="3D Model Printability Checker", font=("Arial", 16))
title.pack(pady=20)

upload_btn = tk.Button(root, text="Upload STL File", command=upload_file)
upload_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=20)

root.mainloop()