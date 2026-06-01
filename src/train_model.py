import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import LinearSVC

df = pd.read_csv(r"C:\Users\ASUS\Desktop\product-category-classifier\data\products.csv")

TITLE_COL = "Product Title"
TARGET_COL = " Category Label"

df = df.dropna(subset=[TITLE_COL, TARGET_COL])

df["title_length"] = df[TITLE_COL].astype(str).apply(len)
df["word_count"] = df[TITLE_COL].astype(str).apply(lambda x: len(x.split()))
df["has_numbers"] = df[TITLE_COL].astype(str).apply(
    lambda x: int(bool(re.search(r"\\d", x)))
)

X = df[[TITLE_COL, "title_length", "word_count", "has_numbers"]]
y = df[TARGET_COL]

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
            TITLE_COL
        ),
        (
            "num",
            StandardScaler(with_mean=False),
            ["title_length", "word_count", "has_numbers"]
        )
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LinearSVC())
])

model.fit(X_train, y_train)

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

joblib.dump(model, "../models/product_classifier.pkl")

print("Model saved successfully!")
