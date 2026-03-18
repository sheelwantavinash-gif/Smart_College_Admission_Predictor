import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Smart Admission Predictor", layout="centered")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("https://images.unsplash.com/photo-1523240795612-9a054b0db644");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    background: rgba(255,255,255,0.05);
    padding: 40px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
}

h1, h2, h3, p, label {
    color: white !important;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
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

st.title("🎓 Smart Admission Predictor")
st.markdown("### Predict your chances instantly")

st.divider()

col1, col2 = st.columns(2)

with col1:
    gre_score = st.slider("Graduate Record Examination Score", 260, 340, 320)
    toefl_score = st.slider("English Language Test Score", 0, 120, 100)
    letter_of_recommendation = st.slider("Letter of Recommendation", 1.0, 5.0, 3.0)
    university_rating = st.slider("University Rating", 1, 5, 3)

with col2:
    cgpa = st.slider("Grade Point Average", 0.0, 10.0, 8.0)
    research = st.selectbox("Research Experience", ["No", "Yes"])
    statement_of_purpose = st.slider("Statement of Purpose", 1.0, 5.0, 3.0)

research_value = 1 if research == "Yes" else 0

st.divider()

if st.button("🚀 Predict"):
    input_data = np.array([[
        gre_score,
        toefl_score,
        university_rating,
        statement_of_purpose,
        letter_of_recommendation,
        cgpa,
        research_value
    ]])

    prediction = model.predict(input_data)[0]

    st.subheader("Result")

    if prediction > 0.7:
        st.success(f"High Chance ({prediction:.2f})")
        st.balloons()

    elif prediction > 0.4:
        st.warning(f"Moderate Chance ({prediction:.2f})")
        st.snow()   # ❄️ SNOW EFFECT HERE

    else:
        st.error(f"Low Chance ({prediction:.2f})")

    st.progress(float(prediction))
