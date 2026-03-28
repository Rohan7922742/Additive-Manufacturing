import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from src.parsing.load_mesh import load_mesh
from src.features.extract_all import extract_features


RAW_DIR = "data/raw"
OUTPUT_FILE = "data/processed/labels.csv"  


def process_file(path):
    try:
        mesh = load_mesh(path)
        features = extract_features(mesh)

        features["file"] = os.path.basename(path)
        features["label"] = os.path.basename(os.path.dirname(path))  
        return features

    except Exception as e:
        print("Failed:", path, "| Error:", e)
        return None


def collect_files():
    stl_files = []

    for root, dirs, files in os.walk(RAW_DIR):   
        for file in files:
            if file.lower().endswith(".stl"):
                stl_files.append(os.path.join(root, file))

    return stl_files


if __name__ == "__main__":

    print("Current working directory:", os.getcwd())
    print("RAW_DIR absolute path:", os.path.abspath(RAW_DIR))
    print("Exists?", os.path.exists(RAW_DIR))

    if os.path.exists(RAW_DIR):
        print("Contents:", os.listdir(RAW_DIR))

    stl_files = collect_files()
    print("Total STL files:", len(stl_files))

    if len(stl_files) == 0:
        print("No STL files found. Check dataset structure.")
        exit()

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_file, stl_files))

    results = [r for r in results if r is not None]

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("Dataset created:", OUTPUT_FILE)