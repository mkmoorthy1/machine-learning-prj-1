import streamlit as st
import pickle
import numpy as np

import matplotlib.pyplot as plt


model = pickle.load(open("health-risk.pkl","rb"))

encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("Health Risk Prediction")

age= st.slider("Age",18,80,22)
diet = st.selectbox("Diet",['Poor','Average','Good'])
exercise_days= st.slider("Exercise Days",0,7,2)
sleep_hours = st.slider("Sleep Hours",3,12,6)
stress = st.selectbox("Stress Level", ['Low', 'Medium', 'High'])
bmi = st.number_input("BMI",15.0,40.0,22.0)
smoking = st.selectbox("Smoking", ['Yes', 'No'])
alcohol = st.selectbox("Alcohol Consumption", ['Low', 'Medium', 'High'])
family_history = st.selectbox("Family History", ['Yes', 'No'])

if st.button("Predict Risk"):
    input_data = [
        age,
        encoders['diet'].transform([diet])[0],
        exercise_days,
        sleep_hours,
        encoders['stress'].transform([stress])[0],
        bmi,
        encoders['smoking'].transform([smoking])[0],
        encoders['alcohol'].transform([alcohol])[0],
        encoders['family_history'].transform([family_history])[0]
    ]

    prediction= model.predict([input_data])

    risk_label = encoders['risk_level'].inverse_transform([prediction[0]])[0]

    factors = {
        
        "Excercise": exercise_days,
        "Sleep": sleep_hours,
        "BMI": bmi,
        
    }

    st.success(risk_label)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(factors.keys(), factors.values(), color='skyblue', edgecolor='black')

    ax.set_title("Health Factors")
    ax.set_xlabel("Factors")
    ax.set_ylabel("Values")
    plt.xticks(rotation=45)

# Show in Streamlit
    st.pyplot(fig)
