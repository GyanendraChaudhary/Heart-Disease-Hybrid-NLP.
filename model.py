import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scipy.sparse import hstack

MODEL_PATH = "model.pkl"

# ---------------- DATA LOADING ----------------
def load_data():
    df = pd.read_csv("data/heart.csv")

    # Create synthetic NLP text
    df["symptoms"] = (
        df["cp"].astype(str) + " chest pain " +
        df["trestbps"].astype(str) + " bp"
    )

    return df


# ---------------- TRAIN MODEL ----------------
def train_model():
    df = load_data()

    X_num = df[["age", "trestbps", "chol", "thalach"]]
    y = df["target"]

    # Scale numeric data
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # NLP Vectorization
    vectorizer = TfidfVectorizer(max_features=100)
    X_text = vectorizer.fit_transform(df["symptoms"])

    # Combine features
    X_final = hstack([X_num_scaled, X_text])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=200, max_depth=8)
    model.fit(X_train, y_train)

    # Accuracy
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # Save
    joblib.dump((model, vectorizer, scaler), MODEL_PATH)

    return model, vectorizer, scaler


# ---------------- LOAD MODEL ----------------
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model, vectorizer, scaler = joblib.load(MODEL_PATH)
    else:
        model, vectorizer, scaler = train_model()

    return model, vectorizer


# ---------------- PREDICTION ----------------
def predict_risk(model, vectorizer, age, bp, chol, hr, cp, text):
    # Prepare numeric data
    num = np.array([[age, bp, chol, hr]])

    # Convert text
    text_vec = vectorizer.transform([text])

    # Combine
    X = hstack([num, text_vec])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1] * 100

    return pred, prob