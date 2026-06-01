from pathlib import Path
import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "products.csv"
MODEL_PATH = BASE_DIR / "models" / "product_classifier.pkl"

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["Product Title", "Category Label"])

df["title_length"] = df["Product Title"].astype(str).apply(len)

df["word_count"] = df["Product Title"].astype(str).apply(
    lambda x: len(x.split())
)

df["has_numbers"] = df["Product Title"].astype(str).apply(
    lambda x: int(bool(re.search(r"\d", x)))
)

X = df[
    [
        "Product Title",
        "title_length",
        "word_count",
        "has_numbers"
    ]
]

y = df["Category Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "text",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50000
            ),
            "Product Title"
        ),
        (
            "num",
            StandardScaler(with_mean=False),
            [
                "title_length",
                "word_count",
                "has_numbers"
            ]
        )
    ]
)

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC())
    ]
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("Model saved successfully!")