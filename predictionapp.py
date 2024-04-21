import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Function to load models
def load_models():
    model = joblib.load('diabetes_model.pkl')
    preprocessor = joblib.load('diabetes_preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_models()

# Main function for the app
def diabetes_app():
    st.title('🩺 Diabetes Prediction Tool')

    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider('Age', min_value=1, max_value=120, value=30, step=1)
        with col2:
            bmi = st.slider('BMI', min_value=10.0, max_value=50.0, value=22.0, step=0.1)
        with col3:
            glucose = st.slider('Glucose Level (mg/dL)', min_value=50, max_value=200, value=99, step=1)

        col4, col5 = st.columns(2)
        with col4:
            hbA1c = st.slider('HbA1c Level (%)', min_value=3.5, max_value=15.0, value=5.5, step=0.1)
        with col5:
            gender = st.radio('Gender', ['Female', 'Male'])

    user_input_df = pd.DataFrame({
        'year': [2022],
        'age': [age],
        'bmi': [bmi],
        'hbA1c_level': [hbA1c],
        'blood_glucose_level': [glucose],
        'gender': [gender],
        'location': ['Alabama'],
        'smoking_history': ['former'],
    })

    diabetes_probability = predict_diabetes(user_input_df)
    type_2_diabetes_probability = diabetes_probability * 0.9
    feedback = provide_feedback(bmi, glucose, hbA1c)

    with st.container():
        st.header('📊 Diabetes Prediction Results')
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader('👤 Your Health Metrics:')
            st.markdown(f"**Age:** {age}")
            st.markdown(f"**BMI:** {bmi} - *{feedback['BMI']}*")
            st.markdown(f"**Glucose Level:** {glucose} mg/dL - *{feedback['Glucose']}*")
            st.markdown(f"**HbA1c Level:** {hbA1c}% - *{feedback['HbA1c']}*")

        with col2:
            st.subheader('🔍 Diabetes Assessment')
            st.metric("Probability of Diabetes", f"{diabetes_probability*100:.2f}%")
            st.metric("Probability of Type 2 Diabetes", f"{type_2_diabetes_probability*100:.2f}%")
            status_color = "🔴" if diabetes_probability > 0.5 else "🟢"
            st.metric("Diagnosis", f"{status_color} {'Diabetic' if diabetes_probability > 0.5 else 'Not Diabetic'}")

    st.header('📈 Visual Health Analysis')
    plot_health_metrics(bmi, glucose, hbA1c)

# Supporting functions used within the app
def predict_diabetes(user_input_df):
    input_transformed = preprocessor.transform(user_input_df)
    prediction_proba = model.predict_proba(input_transformed)
    return prediction_proba[0][1]

def provide_feedback(bmi, glucose, hbA1c):
    feedback = {
        'BMI': "Underweight" if bmi < 18.5 else "Normal weight" if bmi <= 24.9 else "Overweight" if bmi <= 29.9 else "Obese",
        'Glucose': "Low" if glucose < 70 else "Normal" if glucose <= 99 else "High (Prediabetic)" if glucose <= 125 else "Very High (Diabetic)",
        'HbA1c': "Normal" if hbA1c < 5.7 else "Elevated (Prediabetic)" if hbA1c <= 6.4 else "High (Diabetic)"
    }
    return feedback

def plot_health_metrics(bmi, glucose, hbA1c):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    categories = [
        ['Underweight (<18.5)', 'Normal (18.5-24.9)', 'Overweight (25-29.9)', 'Obese (30+)'],
        ['Low (<70)', 'Normal (70-99)', 'High (100-125)', 'Very High (>125)'],
        ['Normal (<5.7)', 'Elevated (5.7-6.4)', 'High (>6.5)']
    ]
    values = [[18.5, 24.9, 29.9, 50], [70, 99, 125, 200], [5.7, 6.4, 15]]
    user_values = [bmi, glucose, hbA1c]
    colors = ['skyblue', 'lightgreen', 'salmon']
    titles = ['BMI', 'Glucose Levels', 'HbA1c Levels']

    for i in range(3):
        ax[i].bar(categories[i], values[i], color=colors[i])
        ax[i].scatter(categories[i], [user_values[i]] * len(categories[i]), color='purple', zorder=5)
        ax[i].set_title(titles[i])

    st.pyplot(fig)

# You can call the app function to test it independently
if __name__ == '__main__':
    diabetes_app()
