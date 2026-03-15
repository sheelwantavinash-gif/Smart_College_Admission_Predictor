import streamlit as st
import pickle
import pandas as pd

# Load the AI Brain
with open(r"C:\Users\Avinash Sheelwant\admission_model.pkl","rb") as file:
    model = pickle.load(file)

# Build the Website UI
st.set_page_config(page_title="Admission Predictor", page_icon="🎓")
st.title("🎓 Smart College Admission Predictor")
st.markdown("Enter your academic profile below to find out your chances of admission!")

col1, col2 = st.columns(2)

with col1:
    gre = st.number_input("GRE Score (out of 340)", min_value=260, max_value=340, value=310)
    toefl = st.number_input("TOEFL Score (out of 120)", min_value=0, max_value=120, value=100)
    rating = st.selectbox("University Rating (1-5)", [1, 2, 3, 4, 5])
    sop = st.slider("Statement of Purpose Strength (1.0 to 5.0)", 1.0, 5.0, 3.0, 0.5)

with col2:
    lor = st.slider("Letter of Recommendation Strength (1.0 to 5.0)", 1.0, 5.0, 3.0, 0.5)
    gpa = st.number_input("Undergrad GPA (out of 10)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
    research = st.radio("Do you have Research Experience?", ["Yes", "No"])

research_val = 1 if research == "Yes" else 0

st.markdown("---")
if st.button("Predict My Chances", type="primary"):
    input_data = pd.DataFrame([[gre, toefl, rating, sop, lor, gpa, research_val]],
         columns=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'GPA', 'Research'])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Congratulations! Your profile is strong. You have a HIGH chance of getting admitted!")
        st.balloons()
    else:
        st.error("⚠️ It might be tough. Consider improving your profile, focusing on research, or applying to safe schools.")        