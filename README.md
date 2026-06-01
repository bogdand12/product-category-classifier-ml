# Product Category Classifier

## Project Description

This project aims to automatically classify products into categories based on their titles.

The dataset contains more than 30,000 products and includes:

- Product Title
- Product ID
- Merchant ID
- Product Code
- Number of Views
- Merchant Rating
- Listing Date
- Category Label

The target variable is Category Label.

---

## Exploratory Data Analysis

The dataset was analysed to:

- identify missing values;
- inspect category distribution;
- analyse title length;
- understand product title structure.

---

## Feature Engineering

Three additional features were created:

### title_length

Number of characters in the product title.

### word_count

Number of words contained in the title.

### has_numbers

Binary feature indicating whether the title contains digits.

In addition, TF-IDF vectorization was applied to product titles.

---

## Model Selection

Three machine learning algorithms were evaluated:

- Multinomial Naive Bayes
- Logistic Regression
- LinearSVC

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

LinearSVC achieved the highest accuracy and best overall classification performance.

Therefore, it was selected as the final model.

---

## Final Model

Pipeline:

TF-IDF Vectorizer + Engineered Features + LinearSVC

---

## Installation

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train_model.py
```

## Predict

```bash
python src/predict_category.py
```

## Example

Input:

iphone 7 32gb gold

Output:

Mobile Phones