from pathlib import Path
import pandas as pd
import joblib
import re

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "product_classifier.pkl"

model = joblib.load(MODEL_PATH)

print("Product Category Predictor")
print("Type 'exit' to stop.\n")

while True:

    title = input("Enter product title: ")

    if title.lower() == "exit":
        break

    sample = pd.DataFrame(
        {
            "Product Title": [title],
            "title_length": [len(title)],
            "word_count": [len(title.split())],
            "has_numbers": [
                int(bool(re.search(r"\d", title)))
            ]
        }
    )

    prediction = model.predict(sample)[0]

    print(f"Predicted category: {prediction}\n")