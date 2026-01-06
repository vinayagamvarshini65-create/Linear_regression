import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline
# Ensure this file exists in your working directory
pipeline = joblib.load("exam_score_pipeline.pkl")

st.title("🎓 Exam Score Prediction")

# User Inputs
age = st.number_input("Age", 10, 60, value=20)
study_hours = st.number_input("Study Hours", 0.0, 24.0, value=5.0)
sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, value=7.0)
attendance = st.number_input("Class Attendance (%)", 0.0, 100.0, value=80.0)

gender = st.selectbox("Gender", ["male", "female", "other"])
course = st.selectbox("Course", ["ba", "b.com", "b.sc", "b.tech", "bba", "bca", "diploma"])
internet_access = st.selectbox("Internet Access", ["yes", "no"])
sleep_quality = st.selectbox("Sleep Quality", ["poor", "average", "good"])
study_method = st.selectbox("Study Method", ["self-study", "group study", "online videos", "coaching"])
facility_rating = st.selectbox("Facility Rating", ["low", "medium", "high"])
exam_difficulty = st.selectbox("Exam Difficulty", ["easy", "moderate", "hard"])

if st.button("Predict"):
    # We must provide EVERY column used during training in the EXACT same order.
    # Based on your CSV, the training features likely included these 12 columns:
    input_df = pd.DataFrame([{
        "student_id": 0,  # Placeholder for ID
        "age": age,
        "gender": gender,
        "course": course,
        "study_hours": study_hours,
        "class_attendance": attendance,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "study_method": study_method,
        "facility_rating": facility_rating,
        "exam_difficulty": exam_difficulty
    }])

    try:
        # Since you used a Pipeline with ColumnTransformer/OneHotEncoder, 
        # you should pass the RAW STRINGS (like 'male' or 'low') directly to the pipeline.
        prediction = pipeline.predict(input_df)
        st.success(f"Predicted Exam Score: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("Debug: Check if your pipeline and input column names match exactly.")
        