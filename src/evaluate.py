import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# بارگذاری مدل
model = joblib.load(MODEL_DIR / "spam_model.pkl")

# خواندن داده
df = pd.read_csv(DATA_DIR / "processed.csv")

X = df["message"]
y = df["label"]

# همان تقسیم‌بندی train.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# پیش‌بینی
predictions = model.predict(X_test)

print("=" * 50)
print("Accuracy")
print("=" * 50)

print(accuracy_score(y_test, predictions))

print("\n")

print("=" * 50)
print("Classification Report")
print("=" * 50)

print(classification_report(y_test, predictions))

print("\n")

print("=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(confusion_matrix(y_test, predictions))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    cmap="Blues"
)

disp.ax_.set_title("Confusion Matrix")

plt.show()