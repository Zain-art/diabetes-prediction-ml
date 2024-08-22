import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('diabetes.pkl', 'rb') as file:
    model = pickle.load(file)

# print('model loaded succfully',model)
# Function to make predictions
def predict_diabetes(inputs):
    # inputs = [float(i) for i in inputs]
    inputs = np.array(inputs).reshape(1, -1)
    try:
        prediction = model.predict(inputs)
        return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
    except AttributeError as e:
        return f'Error: {e}'

# Streamlit app
def main():
    st.title("Diabetes Prediction App")
    st.write("Enter the details to predict if a person has diabetes.")

    # Input fields for user data
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, step=1)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0, max_value=70, step=1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0, max_value=3, step=0)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

    # Button for prediction
    if st.button("Predict"):
        inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
        result = predict_diabetes(inputs)
        st.success(f"The person is {result}")

if __name__ == "__main__":
    main()
