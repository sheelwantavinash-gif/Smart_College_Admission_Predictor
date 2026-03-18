import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Admission Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url("https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.main-container {
    background: rgba(255, 255, 255, 0.08);
    padding: 30px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.3);
}

h1, h2, h3, p, label {
    color: white !important;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.2s;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("admission_model.pkl")

model = load_model()

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🎓 Smart Admission Predictor")
st.markdown("### Predict your chances instantly")

st.divider()

col1, col2 = st.columns(2)

with col1:
    gre_score = st.slider("Graduate Record Examination Score", 260, 340, 300)
    toefl_score = st.slider("English Language Test Score", 0, 120, 100)

with col2:
    cgpa = st.slider("Grade Point Average", 0.0, 10.0, 8.0)
    research = st.selectbox("Research Experience", ["No", "Yes"])

research_value = 1 if research == "Yes" else 0

st.divider()

if st.button("🚀 Predict"):
    input_data = np.array([[gre_score, toefl_score, cgpa, research_value]])
    prediction = model.predict(input_data)[0]

    st.subheader("Result")

    if prediction > 0.7:
        st.success(f"High Chance ({prediction:.2f})")
    elif prediction > 0.4:
        st.warning(f"Moderate Chance ({prediction:.2f})")
    else:
        st.error(f"Low Chance ({prediction:.2f})")

    st.progress(float(prediction))

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:white;'>Made by Avinash Sheelwant 🚀</p>", unsafe_allow_html=True)
