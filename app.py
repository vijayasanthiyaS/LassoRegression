import streamlit as st
import joblib
import numpy as np

model =joblib.load("diabetes_lasso_model.pk1")

st.title("Diabetes Prediction App")

pregancy = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp= st.number_input("BloodPressure")
SkinThickness = st.number_input("SkinThickness")
Insulin = st.number_input("Insulin")
BMI = st.number_input("BMI")
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction")
Age = st.number_input("Age")


if st.button("Predict"):
    data = np.array([[pregancy,glucose,bp,SkinThickness,Insulin ,BMI,DiabetesPedigreeFunction,Age]])

    pred = model.predict(data)

    if pred > 0.5:
        st.error("Diabetes Detected")
    else:
        st.success("No Diabetes")

# stock prediction, marketing analytics, medical prediction, 