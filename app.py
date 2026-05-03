import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Heart Disease Prediction using Hybrid NLP",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Dataset
# -----------------------------
data = [
    [63, 145, 233, 150, 1, "chest pain fatigue high blood pressure", 1],
    [67, 160, 286, 108, 1, "chest pressure sweating shortness of breath", 1],
    [57, 150, 276, 112, 1, "left arm pain chest discomfort dizziness", 1],
    [64, 170, 303, 122, 1, "severe chest pain high bp sweating", 1],
    [71, 155, 265, 105, 1, "irregular heartbeat chest heaviness fatigue", 1],
    [59, 148, 254, 120, 1, "breathlessness chest tightness nausea", 1],
    [62, 162, 295, 115, 1, "jaw pain cold sweat chest pressure", 1],
    [69, 158, 310, 100, 1, "palpitations chest pain weakness", 1],
    [55, 142, 260, 118, 1, "diabetes high cholesterol chest pain", 1],
    [73, 165, 330, 98, 1, "shortness of breath sweating chest pain", 1],

    [28, 118, 180, 170, 0, "normal breathing active lifestyle", 0],
    [35, 120, 190, 165, 0, "mild headache no chest pain", 0],
    [42, 125, 200, 160, 0, "regular exercise no fatigue", 0],
    [31, 115, 175, 172, 0, "slight fever cough no heart pain", 0],
    [45, 128, 210, 155, 0, "normal cholesterol no breathing issue", 0],
    [38, 122, 185, 168, 0, "minor stomach pain no chest discomfort", 0],
    [50, 130, 215, 150, 0, "good sleep normal heart rate", 0],
    [33, 119, 178, 176, 0, "no sweating no chest pain", 0],
    [47, 126, 205, 158, 0, "normal activity no dizziness", 0],
    [40, 121, 195, 162, 0, "light cough no chest pain", 0],
]

df = pd.DataFrame(
    data,
    columns=["age", "blood_pressure", "cholesterol", "max_heart_rate", "chest_pain", "symptoms", "target"]
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["symptoms"] = df["symptoms"].apply(clean_text)

# -----------------------------
# Model Pipeline
# -----------------------------
X = df[["age", "blood_pressure", "cholesterol", "max_heart_rate", "chest_pain", "symptoms"]]
y = df["target"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "blood_pressure", "cholesterol", "max_heart_rate", "chest_pain"]),
        ("text", TfidfVectorizer(ngram_range=(1, 2)), "symptoms")
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=120, random_state=42))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Medical Keyword Engine
# -----------------------------
risk_keywords = {
    "chest pain": 18,
    "chest pressure": 18,
    "chest tightness": 16,
    "shortness of breath": 18,
    "breathlessness": 18,
    "sweating": 12,
    "cold sweat": 14,
    "left arm pain": 15,
    "jaw pain": 13,
    "dizziness": 9,
    "fatigue": 8,
    "high bp": 12,
    "high blood pressure": 12,
    "diabetes": 12,
    "high cholesterol": 12,
    "palpitations": 14,
    "irregular heartbeat": 16,
    "nausea": 8,
    "weakness": 8
}

def keyword_risk_score(text):
    text = clean_text(text)
    score = 0
    found = []

    for keyword, value in risk_keywords.items():
        if keyword in text:
            score += value
            found.append(keyword)

    return min(score, 100), found

def clinical_score(age, bp, cholesterol, heart_rate, chest_pain):
    score = 0

    if age >= 60:
        score += 18
    elif age >= 45:
        score += 10

    if bp >= 160:
        score += 18
    elif bp >= 140:
        score += 12

    if cholesterol >= 300:
        score += 18
    elif cholesterol >= 240:
        score += 12

    if heart_rate < 110:
        score += 12
    elif heart_rate < 130:
        score += 6

    if chest_pain == 1:
        score += 20

    return min(score, 100)

def predict_advanced(age, bp, cholesterol, heart_rate, chest_pain, symptoms):
    input_df = pd.DataFrame({
        "age": [age],
        "blood_pressure": [bp],
        "cholesterol": [cholesterol],
        "max_heart_rate": [heart_rate],
        "chest_pain": [chest_pain],
        "symptoms": [clean_text(symptoms)]
    })

    ml_probability = model.predict_proba(input_df)[0][1] * 100
    keyword_score, matched_keywords = keyword_risk_score(symptoms)
    clinical = clinical_score(age, bp, cholesterol, heart_rate, chest_pain)

    final_score = (0.50 * ml_probability) + (0.25 * keyword_score) + (0.25 * clinical)

    if final_score >= 70:
        risk = "High Risk"
    elif final_score >= 40:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    reasons = []

    if age >= 60:
        reasons.append("Age is high")
    if bp >= 140:
        reasons.append("Blood pressure is above normal range")
    if cholesterol >= 240:
        reasons.append("Cholesterol level is high")
    if heart_rate < 130:
        reasons.append("Max heart rate is relatively low")
    if chest_pain == 1:
        reasons.append("Chest pain is present")
    for word in matched_keywords:
        reasons.append(f"Symptom detected: {word}")

    return {
        "risk": risk,
        "final_score": round(final_score, 2),
        "ml_probability": round(ml_probability, 2),
        "keyword_score": round(keyword_score, 2),
        "clinical_score": round(clinical, 2),
        "matched_keywords": matched_keywords,
        "reasons": reasons
    }

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
    color: #ff4b5c;
}
.sub-title {
    text-align: center;
    color: #94a3b8;
    font-size: 18px;
}
.card {
    padding: 22px;
    border-radius: 18px;
    background-color: #111827;
    border: 1px solid #374151;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Prediction", "Model Performance", "Dataset", "Methodology", "About"]
)

# -----------------------------
# Home
# -----------------------------
if page == "Home":
    st.markdown('<div class="main-title">❤️ Heart Disease Prediction using Hybrid NLP</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Advanced Final Year Project | NLP + Clinical Data + Machine Learning + Explainable AI</div>', unsafe_allow_html=True)
    st.write("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frontend", "Streamlit")
    c2.metric("Backend", "Python")
    c3.metric("Model", "Random Forest")
    c4.metric("Accuracy", f"{round(accuracy * 100, 2)}%")

    st.subheader("Project Overview")
    st.write("""
    This system predicts possible heart disease risk using a hybrid approach. 
    It combines structured clinical parameters such as age, blood pressure, cholesterol and heart rate 
    with unstructured symptom text processed through Natural Language Processing.
    """)

    st.subheader("Why this project is advanced?")
    st.write("""
    Unlike simple heart disease prediction systems, this project uses:
    - Clinical data analysis
    - NLP based symptom understanding
    - Machine learning classification
    - Medical keyword risk scoring
    - Explainable prediction reasons
    - Downloadable patient report
    """)

# -----------------------------
# Prediction
# -----------------------------
elif page == "Prediction":
    st.title("Advanced Patient Risk Prediction")

    col1, col2 = st.columns(2)

    with col1:
        patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
        age = st.slider("Age", 18, 90, 55)
        bp = st.slider("Blood Pressure", 80, 220, 140)
        cholesterol = st.slider("Cholesterol", 120, 400, 240)

    with col2:
        heart_rate = st.slider("Max Heart Rate", 70, 210, 140)
        chest_pain_option = st.selectbox("Chest Pain Present?", ["No", "Yes"])
        chest_pain = 1 if chest_pain_option == "Yes" else 0

        symptoms = st.text_area(
            "Enter Symptoms",
            placeholder="Example: chest pain, sweating, shortness of breath, fatigue",
            height=140
        )

    if st.button("Predict Risk"):
        if symptoms.strip() == "":
            st.warning("Please enter symptoms.")
        else:
            result = predict_advanced(age, bp, cholesterol, heart_rate, chest_pain, symptoms)

            st.write("---")
            st.subheader("Prediction Result")

            if result["risk"] == "High Risk":
                st.error(f"⚠️ {result['risk']}")
            elif result["risk"] == "Medium Risk":
                st.warning(f"🟡 {result['risk']}")
            else:
                st.success(f"✅ {result['risk']}")

            st.progress(int(result["final_score"]) / 100)

            a, b, c, d = st.columns(4)
            a.metric("Final Risk Score", f"{result['final_score']}%")
            b.metric("ML Score", f"{result['ml_probability']}%")
            c.metric("NLP Keyword Score", f"{result['keyword_score']}%")
            d.metric("Clinical Score", f"{result['clinical_score']}%")

            st.subheader("Why this prediction?")
            if result["reasons"]:
                for reason in result["reasons"]:
                    st.write(f"• {reason}")
            else:
                st.write("No major risk reason detected.")

            st.subheader("Risk Score Breakdown")
            labels = ["ML Score", "NLP Score", "Clinical Score", "Final Score"]
            values = [
                result["ml_probability"],
                result["keyword_score"],
                result["clinical_score"],
                result["final_score"]
            ]

            fig, ax = plt.subplots()
            ax.bar(labels, values)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Score (%)")
            ax.set_title("Hybrid Risk Score Breakdown")
            st.pyplot(fig)

            st.subheader("Health Recommendation")
            if result["risk"] == "High Risk":
                st.write("Immediate medical consultation is recommended. Patient should consult a cardiologist.")
            elif result["risk"] == "Medium Risk":
                st.write("Patient should monitor symptoms and go for a medical checkup.")
            else:
                st.write("Risk appears low, but regular health monitoring is recommended.")

            report = f"""
Heart Disease Prediction Report

Patient Name: {patient_name if patient_name else "Not Provided"}
Age: {age}
Blood Pressure: {bp}
Cholesterol: {cholesterol}
Max Heart Rate: {heart_rate}
Chest Pain: {chest_pain_option}

Symptoms:
{symptoms}

Prediction Result: {result['risk']}
Final Risk Score: {result['final_score']}%
ML Probability Score: {result['ml_probability']}%
NLP Keyword Score: {result['keyword_score']}%
Clinical Score: {result['clinical_score']}%

Prediction Reasons:
{chr(10).join(result['reasons']) if result['reasons'] else "No major reason detected"}

Disclaimer:
This project is for educational purposes only and should not be used for real medical diagnosis.
"""
            st.download_button(
                "Download Patient Report",
                report,
                file_name="heart_disease_prediction_report.txt"
            )

# -----------------------------
# Model Performance
# -----------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    st.metric("Accuracy", f"{round(accuracy * 100, 2)}%")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Low Risk", "High Risk"])
    ax.set_yticklabels(["Low Risk", "High Risk"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# -----------------------------
# Dataset
# -----------------------------
elif page == "Dataset":
    st.title("Dataset")
    st.write("This dataset contains both clinical values and symptom text.")
    st.dataframe(df)

    st.write("Target 0 = Low Risk")
    st.write("Target 1 = High Risk")

# -----------------------------
# Methodology
# -----------------------------
elif page == "Methodology":
    st.title("Proposed Methodology")

    st.write("""
    The proposed system uses a hybrid approach consisting of three major parts:
    """)

    st.subheader("1. Clinical Feature Analysis")
    st.write("Age, blood pressure, cholesterol, heart rate and chest pain status are analyzed.")

    st.subheader("2. NLP Symptom Analysis")
    st.write("Patient symptoms are cleaned and converted into numerical form using TF-IDF vectorization.")

    st.subheader("3. Machine Learning Classification")
    st.write("Random Forest Classifier is used to predict heart disease risk.")

    st.subheader("4. Hybrid Decision Engine")
    st.code("Final Score = 50% ML Score + 25% NLP Keyword Score + 25% Clinical Score")

    st.subheader("System Architecture")
    st.code("""
Patient Input
↓
Clinical Data + Symptom Text
↓
NLP Preprocessing
↓
TF-IDF Vectorization
↓
Random Forest ML Model
↓
Keyword Risk Engine
↓
Hybrid Score Calculation
↓
Risk Prediction + Report
""")

# -----------------------------
# About
# -----------------------------
elif page == "About":
    st.title("About Project")

    st.subheader("Project Title")
    st.write("Heart Disease Prediction using Hybrid NLP and Machine Learning")

    st.subheader("Frontend")
    st.write("""
    Streamlit based interactive web application with multi-page dashboard, prediction form, graphs and report download.
    """)

    st.subheader("Backend")
    st.write("""
    Python backend with NLP preprocessing, TF-IDF feature extraction, Random Forest classification and hybrid risk scoring.
    """)

    st.subheader("Technologies Used")
    st.write("""
    Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, NLP, TF-IDF, Random Forest
    """)

    st.warning("This project is for educational purposes only and is not a substitute for real medical diagnosis.")