import streamlit as st
import numpy as np
import pickle

# Load the model
with open('Diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Diabetes Prediction")

# Get user input
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0)
insulin = st.number_input("Insulin", min_value=0.0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

# Prepare the input data
input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, 
                       insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)

# Predict and display the result
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_prob = model.predict_proba(input_data)[0][1]  # Probability of positive class
    
    if prediction[0] == 1:
        st.write("Predicted Outcome: Diabetes Positive")
        st.write(f"Probability: {prediction_prob:.2f}")
        st.write("### Suggestions:")
        st.write("1. Consult a healthcare provider for further testing.")
        st.write("2. Consider lifestyle changes such as diet and exercise.")
        st.write("3. Regular monitoring of blood sugar levels.")
    else:
        st.write("Predicted Outcome: Diabetes Negative")
        st.write(f"Probability: {prediction_prob:.2f}")
        st.write("### Suggestions:")
        st.write("1. Maintain a healthy lifestyle with a balanced diet and regular exercise.")
        st.write("2. Continue regular health check-ups.")
