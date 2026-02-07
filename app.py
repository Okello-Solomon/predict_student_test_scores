import streamlit as st
import pandas as pd 
import numpy as np 
import joblib 


# Load the trained model
model = joblib.load('linear_regression.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('le.pkl')
feature_names = joblib.load('feature_names.pkl')

# Set up the streamlit app
st.title('Predicting Student Test Scores')
st.write('Predict Test Scores Based on Student Details')
st.subheader('Enter Student Information Below to Generate a Prediction')

#Create input fields for the user to enter students details
age = st.number_input("Age", min_value=15, max_value=30, value=20)

gender = st.selectbox(
    "Gender",
    ['female', 'male', 'other']
)

course = st.selectbox(
    "Course",
    ['b.sc', 'diploma', 'bca', 'b.com', 'ba', 'bba', 'b.tech']
)

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0, max_value=12.0, value=4.0
)

class_attendance = st.number_input(
    "Class Attendance (%)",
    min_value=0.0, max_value=100.0, value=80.0
)

internet_access = st.selectbox(
    "Internet Access",
    ['no', 'yes']
)

sleep_hours = st.number_input(
    "Sleep Hours per Day",
    min_value=0.0, max_value=12.0, value=7.0
)

sleep_quality = st.selectbox(
    "Sleep Quality",
    ['poor', 'average', 'good']
)

study_method = st.selectbox(
    "Study Method",
    ['online videos', 'self-study', 'coaching', 'group study', 'mixed']
)

facility_rating = st.selectbox(
    "Facility Rating",
    ['low', 'medium', 'high']
)

exam_difficulty = st.selectbox(
    "Exam Difficulty",
    ['easy', 'moderate', 'hard']
)

#Create button to make prediction

if st.button("Predict Test Score"):
    # encode the input data
    internet_access=le.transform([internet_access])[0]

    # dummy encode
    gender_male = 1 if gender == 'female' else 0
    gender_other = 1 if gender == 'other' else 0
    #gender_male is the reference category, so dont need to create a dummy variable for is male -> female=0

    course_b_sc = 1 if course == 'b.sc' else 0
    course_b_tech = 1 if course == 'b.tech' else 0
    course_ba = 1 if course == 'ba' else 0
    course_bba = 1 if course == 'bba' else 0
    course_bca = 1 if course == 'bca' else 0
    course_diploma = 1 if course == 'diploma' else 0
    # course_b.com is reference, so no dummy variable

    study_method_group_study = 1 if study_method == 'group study' else 0
    study_method_mixed = 1 if study_method == 'mixed' else 0
    study_method_online_videos = 1 if study_method == 'online videos' else 0
    study_method_self_study = 1 if study_method == 'self-study' else 0
    # study_method_self_study is reference

    # Ordinal
    sleep_quality_map = {'poor': 0, 'average': 1, 'good': 2}
    facility_rating_map = {'low': 0, 'medium': 1, 'high': 2}
    exam_difficulty_map = {'easy': 0, 'moderate': 1, 'hard': 2}

    sleep_quality = sleep_quality_map[sleep_quality]
    facility_rating = facility_rating_map[facility_rating]
    exam_difficulty = exam_difficulty_map[exam_difficulty]

    # Create a Dataframe for the input data
    input_data = pd.DataFrame([[
    age,
    gender_male,
    gender_other,
    course_b_sc,
    course_b_tech,
    course_ba,
    course_bba,
    course_bca,
    course_diploma,
    study_hours,
    class_attendance,
    internet_access,
    sleep_hours,
    sleep_quality,
    study_method_group_study,
    study_method_mixed,
    study_method_online_videos,
    study_method_self_study,
    facility_rating,
    exam_difficulty
]], columns=feature_names)


  #Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f"**Predicted Student Test Score:** {prediction[0]:.2f}")
