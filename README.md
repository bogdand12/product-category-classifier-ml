# Product Category Classifier

Machine Learning project for automatic product category classification using product titles and additional engineered features.

## Project Structure

```bash
product-category-classifier/
│
├── data/
│   └── products.csv
│
├── models/
│   └── product_classifier.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── train_model.py
│   └── predict_category.py
│
├── requirements.txt
└── README.md
```

## Model Used

- TF-IDF Vectorizer
- LinearSVC Classifier

## Feature Engineering

Additional features:
- title_length
- word_count
- has_numbers

## Model Accuracy

Accuracy achieved on test set:

**0.9604**

## Installation

```bash
pip install -r requirements.txt
```

## Train Model

```bash
cd src
python train_model.py
```

## Predict Category

```bash
cd src
python predict_category.py
```

## Example Inputs

| Product Title | Expected Category |
|---|---|
| iphone 7 32gb gold | Mobile Phones |
| olympus e m10 mark iii geh use silber | Digital Cameras |
| kenwood k20mss15 solo | Microwaves |

## Technologies

- Python
- Pandas
- Scikit-learn
- Jupyter Notebook

