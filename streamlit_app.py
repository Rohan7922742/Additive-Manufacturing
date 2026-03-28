import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import os
import sys
import pyvista as pv
import tempfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

from src.parsing.load_mesh import load_mesh
from src.features.extract_all import extract_features

MODEL_PATH = os.path.join(PROJECT_ROOT, "saved_models", "model.pkl")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

root = tk.Tk()
root.title("3D Printability Predictor")
root.geometry("600x500")
root.configure(bg="#1e1e2f") 

LABEL_FONT = ("Helvetica", 12, "bold")
BUTTON_FONT = ("Helvetica", 11, "bold")
RESULT_FONT = ("Helvetica", 12, "bold")
BG_COLOR = "#1e1e2f"
FG_COLOR = "#ffffff"
BUTTON_COLOR = "#4a90e2"
ENTRY_COLOR = "#2c2c3f"

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("STL files", "*.stl")])
    if file_path:
        entry_file.delete(0, tk.END)
        entry_file.insert(0, file_path)

def predict():
    file_path = entry_file.get()
    if not file_path:
        messagebox.showerror("Error", "Please select an STL file")
        return

    try:
        mesh = load_mesh(file_path)
        features = extract_features(mesh)
        X = pd.DataFrame([features])[feature_columns]

        probs = model.predict_proba(X)[0]
        pred_index = probs.argmax()
        prediction = model.classes_[pred_index]

        label_result.config(text=f"Prediction: {prediction}")
        label_probs.config(text=f"Probabilities → Printable: {probs[0]:.2f}, Not Printable: {probs[1]:.2f}")

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(pv.read(file_path), color="#4a90e2")
        screenshot_path = os.path.join(tempfile.gettempdir(), "mesh.png")
        plotter.show(screenshot=screenshot_path)
        img = tk.PhotoImage(file=screenshot_path)
        label_image.config(image=img)
        label_image.image = img

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process STL file:\n{e}")

frame_top = tk.Frame(root, bg=BG_COLOR)
frame_top.pack(pady=20)

frame_middle = tk.Frame(root, bg=BG_COLOR)
frame_middle.pack(pady=10)

frame_bottom = tk.Frame(root, bg=BG_COLOR)
frame_bottom.pack(pady=20)

tk.Label(frame_top, text="3D Printability Predictor", font=("Helvetica", 18, "bold"), bg=BG_COLOR, fg="#4a90e2").pack()
tk.Label(frame_top, text="Upload an STL file to check if it's printable", font=("Helvetica", 12), bg=BG_COLOR, fg=FG_COLOR).pack()

tk.Label(frame_middle, text="Select STL File:", font=LABEL_FONT, bg=BG_COLOR, fg=FG_COLOR).grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_file = tk.Entry(frame_middle, width=40, bg=ENTRY_COLOR, fg=FG_COLOR, insertbackground=FG_COLOR)
entry_file.grid(row=0, column=1, padx=5, pady=5)
tk.Button(frame_middle, text="Browse", command=browse_file, font=BUTTON_FONT, bg=BUTTON_COLOR, fg=FG_COLOR).grid(row=0, column=2, padx=5, pady=5)
tk.Button(frame_middle, text="Predict", command=predict, font=BUTTON_FONT, bg=BUTTON_COLOR, fg=FG_COLOR, width=15).grid(row=1, column=0, columnspan=3, pady=10)

label_result = tk.Label(frame_bottom, text="Prediction: ", font=RESULT_FONT, bg=BG_COLOR, fg="#f39c12")
label_result.pack(pady=5)

label_probs = tk.Label(frame_bottom, text="Probabilities: ", font=RESULT_FONT, bg=BG_COLOR, fg="#f39c12")
label_probs.pack(pady=5)

label_image = tk.Label(frame_bottom, bg=BG_COLOR)
label_image.pack(pady=10)

root.mainloop()
