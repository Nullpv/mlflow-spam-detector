# 📩 Spam SMS Detection with Machine Learning, MLflow & FastAPI

A complete end-to-end Machine Learning project for SMS Spam Detection using Python.

This project demonstrates the complete ML workflow from data preprocessing to model deployment through a REST API.

---

# 🚀 Features

- Data preprocessing
- SMS spam classification
- TF-IDF feature extraction
- Naive Bayes classifier
- Logistic Regression classifier
- Model evaluation
- MLflow experiment tracking
- Model serialization with Joblib
- REST API using FastAPI
- Interactive Swagger documentation

---

# 📂 Project Structure

```
mlflow-spam-detector/
│
├── data/
│   ├── spam.csv
│   └── processed.csv
│
├── models/
│   ├── naive_bayes.pkl
│   └── logistic_regression.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── app.py
│   └── check_model.py
│
├── mlruns/
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Machine Learning Pipeline

Dataset

↓

Data Cleaning

↓

Label Encoding

↓

Train/Test Split

↓

TF-IDF Vectorization

↓

Naive Bayes

↓

Logistic Regression

↓

Model Evaluation

↓

MLflow Tracking

↓

Joblib Model Saving

↓

FastAPI Deployment

---

# 📊 Dataset

Dataset:

SMS Spam Collection Dataset

Classes

- Ham
- Spam

Features

- message
- label

---

# ⚙️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- FastAPI
- Uvicorn
- Joblib

---

# 📈 Evaluation Metrics

The following metrics are calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Example:

Naive Bayes

Accuracy: 95.84%

Logistic Regression

Accuracy: 96.71%

---

# ▶️ Installation

Clone the project

```bash
git clone https://github.com/yourusername/mlflow-spam-detector.git

cd mlflow-spam-detector
```

Create virtual environment

```bash
python -m venv env
```

Linux

```bash
source env/bin/activate
```

Windows

```bash
env\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Train Model

```bash
python src/train.py
```

---

# 🔍 Evaluate Model

```bash
python src/evaluate.py
```

---

# 💬 Predict from Terminal

```bash
python src/predict.py
```

Example

```
Enter message:

Congratulations!
You won a FREE iPhone.

Prediction:

SPAM
```

---

# 🌐 Run FastAPI

```bash
uvicorn src.app:app --reload
```

API

```
http://127.0.0.1:8000
```

Swagger

```
http://127.0.0.1:8000/docs
```

---

# 📦 MLflow

Run MLflow UI

```bash
mlflow ui
```

Open

```
http://127.0.0.1:5000
```

---

# 📌 Models

## Naive Bayes

Vectorizer

TF-IDF

Classifier

Multinomial Naive Bayes

---

## Logistic Regression

Vectorizer

TF-IDF

Classifier

Logistic Regression

---

# 📈 Example API Request

```json
{
  "message": "URGENT! You have won a FREE membership."
}
```

Response

```json
{
  "prediction": "SPAM"
}
```

---

# 🔮 Future Improvements

- Docker support
- CI/CD Pipeline
- Kubernetes Deployment
- Cloud Deployment
- Model Registry
- Hyperparameter Tuning
- Deep Learning Models
- Transformer-based Spam Detection

---

# 👨‍💻 Author

Saeed Solgi

Master Student of Software Engineering

Machine Learning | Backend Development | MLOps
