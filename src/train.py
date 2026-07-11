from pathlib import Path
import os
import joblib
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ===============================
# Paths
# ===============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("=" * 50)
print("Current Working Directory")
print(os.getcwd())
print("=" * 50)

# ===============================
# Load Dataset
# ===============================

df = pd.read_csv(DATA_DIR / "processed.csv")

print(df.head())

X = df["message"]
y = df["label"]

# ===============================
# Train / Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train:", len(X_train))
print("Test :", len(X_test))

# ===============================
# Pipelines
# ===============================

pipeline_nb = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

pipeline_lr = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# ===============================
# MLflow Experiment
# ===============================

mlflow.set_experiment("Spam Detection")

# ============================================================
# Naive Bayes
# ============================================================

with mlflow.start_run(run_name="Naive Bayes"):

    pipeline_nb.fit(X_train, y_train)

    predictions = pipeline_nb.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n========== Naive Bayes ==========")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("vectorizer", "TF-IDF")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=pipeline_nb,
        name="naive_bayes_model"
    )

    joblib.dump(
        pipeline_nb,
        MODEL_DIR / "naive_bayes.pkl"
    )

# ============================================================
# Logistic Regression
# ============================================================

with mlflow.start_run(run_name="Logistic Regression"):

    pipeline_lr.fit(X_train, y_train)

    predictions = pipeline_lr.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print("\n====== Logistic Regression ======")
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("vectorizer", "TF-IDF")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(
        sk_model=pipeline_lr,
        name="logistic_regression_model"
    )

    joblib.dump(
        pipeline_lr,
        MODEL_DIR / "logistic_regression.pkl"
    )

print("\n" + "=" * 50)
print("Training Finished Successfully")
print("=" * 50)
