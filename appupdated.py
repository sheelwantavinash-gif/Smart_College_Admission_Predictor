import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Admission Predictor", page_icon="🎓", layout="centered")

def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1523240795612-9a054b0db644");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

@st.cache_resource
def load_model():
    return joblib.load("admission_model.pkl")

model = load_model()

st.title("🎓 Admission Predictor")
st.markdown("Enter your academic details to estimate your admission chances.")

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

if st.button("Predict Admission Chance"):
    input_data = np.array([[gre_score, toefl_score, cgpa, research_value]])
    prediction = model.predict(input_data)[0]

    if prediction > 0.7:
        st.success(f"High Chance ({prediction:.2f})")
    elif prediction > 0.4:
        st.warning(f"Moderate Chance ({prediction:.2f})")
    else:
        st.error(f"Low Chance ({prediction:.2f})")

    st.progress(float(prediction))

    data = pd.DataFrame({
        "Feature": ["GRE", "TOEFL", "CGPA", "Research"],
        "Value": [gre_score, toefl_score, cgpa, research_value]
    })

    fig, ax = plt.subplots()
    ax.bar(data["Feature"], data["Value"])
    st.pyplot(fig)

st.markdown("---")
st.markdown("Developed by Avinash Sheelwant")