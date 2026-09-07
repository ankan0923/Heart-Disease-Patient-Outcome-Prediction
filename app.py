from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Heart Disease Risk Explorer", page_icon="❤️", layout="wide")
BASE = Path(__file__).parent
DATA_FILE = BASE / "heart_disease_cleveland.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["heart_disease"] = (df.pop("diagnosis") > 0).astype(int)
    return df

@st.cache_resource
def train_model(df):
    categorical = ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg",
                   "exercise_angina", "st_slope", "thalassemia"]
    numeric = [c for c in df.columns if c not in categorical + ["heart_disease"]]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("preprocessor", prep),
                      ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    model.fit(df.drop(columns="heart_disease"), df["heart_disease"])
    return model

df = load_data()
model = train_model(df)

st.title("❤️ Heart Disease Risk Explorer")
st.caption("Portfolio Dashboard — Just for Diagnosis or Treatment Demonstration")

page = st.sidebar.selectbox(
    "Dashboard Sections",
    ["Overview", "Patient Risk Explorer", "Feature Insights"])

# Show content based on selection
if page == "Overview":
    st.write("📊 Welcome to the Heart Disease Risk Explorer!")
elif page == "Patient Risk Explorer":
    st.write("🧑‍⚕️ Explore patient risk factors here.")
elif page == "Feature Insights":
    st.write("🔍 See feature importance and insights.")

if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Patients", f"{len(df):,}")
    c2.metric("Observed disease cases", int(df["heart_disease"].sum()))
    c3.metric("Sample prevalence", f"{df['heart_disease'].mean():.1%}")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="age", color="heart_disease", barmode="overlay",
                           labels={"heart_disease":"Outcome"}, title="Age Distribution by Outcome")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        summary = df["heart_disease"].value_counts().rename(index={0:"No disease", 1:"Disease"})
        fig = px.pie(values=summary.values, names=summary.index, hole=.55,
                     title="Outcome Mix")
        st.plotly_chart(fig, use_container_width=True)
    st.info("The data contains 303 Cleveland Clinic records. Treat all findings as exploratory because the sample is small and historical.")

elif page == "Patient Risk Explorer":
    st.subheader("Enter Patient Measurements")
    with st.form("patient_form"):
        a, b, c = st.columns(3)
        age = a.slider("Age", 20, 90, 55)
        sex = a.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chest = a.selectbox("Chest pain type", [1, 2, 3, 4])
        resting_bp = b.slider("Resting blood pressure", 80, 220, 130)
        chol = b.slider("Cholesterol", 100, 600, 240)
        fbs = b.selectbox("Fasting blood sugar >120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ecg = c.selectbox("Resting ECG code", [0, 1, 2])
        max_hr = c.slider("Maximum heart rate", 60, 220, 150)
        exang = c.selectbox("Exercise-induced angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        d, e, f = st.columns(3)
        oldpeak = d.slider("ST depression", 0.0, 7.0, 1.0, 0.1)
        slope = e.selectbox("ST slope code", [1, 2, 3])
        ca = e.selectbox("Major vessels", [0, 1, 2, 3])
        thal = f.selectbox("Thalassemia code", [3, 6, 7])
        submitted = st.form_submit_button("Estimate model probability")
    if submitted:
        patient = pd.DataFrame([{"age":age, "sex":sex, "chest_pain_type":chest,
            "resting_bp":resting_bp, "cholesterol":chol, "fasting_blood_sugar":fbs,
            "resting_ecg":ecg, "max_heart_rate":max_hr, "exercise_angina":exang,
            "st_depression":oldpeak, "st_slope":slope, "major_vessels":ca,
            "thalassemia":thal}])
        probability = model.predict_proba(patient)[0, 1]
        st.metric("Estimated probability of the positive class", f"{probability:.1%}")
        st.progress(float(probability))
        st.warning("This output is a model demonstration, not medical advice. A clinician must interpret real patient findings.")

else:
    outcome = st.selectbox("Outcome group", ["All", "No disease", "Disease"])
    view = df if outcome == "All" else df[df["heart_disease"] == (outcome == "Disease")]
    x = st.selectbox("X-axis", ["age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression"])
    y = st.selectbox("Y-axis", ["max_heart_rate", "cholesterol", "resting_bp", "age", "st_depression"])
    fig = px.scatter(view, x=x, y=y, color="heart_disease", hover_data=["chest_pain_type", "exercise_angina"],
                     title=f"{x.replace('_',' ').title()} vs {y.replace('_',' ').title()}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(view.groupby("heart_disease")[["age", "resting_bp", "cholesterol", "max_heart_rate", "st_depression"]].mean().round(1), use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Source: UCI Heart Disease — Cleveland subset")
st.sidebar.caption("Design By - Ankan Chowdhury")



