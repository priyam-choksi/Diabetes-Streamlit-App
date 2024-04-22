import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from itertools import combinations
import joblib

##Priyam Deepak Choksi - 21-04-2024

@st.cache_data
def project_description():
    st.title("Diabetes Prediction and Analysis Web Application")
    st.write("""
---
Project Video : 
1. [Project Video uploaded on YouTube](https://youtu.be/gIcCXWj7e-A)
---
#### Project Overview
This project aims to assess the likelihood of diabetes based on various health metrics provided by the user. The application leverages a Logistic Regression model, well-suited for binary classification tasks, to predict the onset of diabetes. It features a user-friendly web interface developed with Streamlit, enabling easy interaction and real-time prediction capabilities.

- **Project Description**: Details the methodologies, data descriptions, and objectives of the project.
- **Prediction Model**: Utilizes a machine learning model to predict diabetes based on input features.
- **Exploratory Data Analysis (EDA)**: This section provides insights through statistical summaries and visualizations.

## Methodologies Used
- **Data Cleaning**: Preprocessing the raw data to handle missing values, outliers, and inconsistencies.
- **Feature Engineering**: Creating new features or transforming existing ones to improve the predictive power of the model.
- **Predictive Modeling with Logistic Regression**: Utilizing a logistic regression model for binary classification to predict the likelihood of diabetes based on health records.
- **Exploratory Data Analysis (EDA)**: Analyzing and visualizing the data to understand patterns, trends, and relationships among variables.
- **Machine Learning Model Evaluation**: Assessing model performance using metrics such as accuracy, precision, recall, and ROC AUC.             

#### Features
1. **User-Friendly Interface**: Utilizes Streamlit to provide a simple, intuitive UI for inputting health metrics.
2. **Immediate Predictions**: Delivers real-time diabetes predictions and Type 2 diabetes probabilities based on user inputs.
3. **Visual Health Metrics**: Graphically displays user health metrics against standard ranges, offering immediate visual feedback.
4. **Exploratory Data Analysis (EDA)**: Includes a section for statistical summaries and visualizations of the underlying data used for model training.

#### How It Works
1. **Enter Health Information**: Input your health metrics using interactive sliders for Age, BMI, Glucose Level, and HbA1c Level.
2. **Receive Predictions**: The app calculates and displays your diabetes probability and where your health metrics stand in comparison to normal ranges.
3. **Explore Underlying Data**: Navigate to the EDA tab to view more data visualizations that provide insights into the dataset.

#### Built With
- **Python**: Primary programming language.
- **Streamlit**: App framework for creating the web interface.
- **Pandas & Numpy**: For data manipulation and numerical calculations.
- **Matplotlib & Plotly**: For generating interactive visualizations.
- **Scikit-Learn**: Utilized for Logistic Regression model implementation.
             
#### Model Information
- **Model Used**: Logistic Regression, which is effective for binary classification problems.
- **Performance**:
  - **Accuracy**: 0.9597
  - **ROC AUC**: 0.9587
- **Model Training Details**:
  - **Max Iterations**: Set to 1000 to ensure convergence.
  - **Features Used**: Includes age, BMI, glucose levels, HbA1c levels, gender, and smoking history.
  - **Coefficient Analysis**: Provided to understand the influence of each feature on the model's predictions.

#### References
- **CDC Diabetes Surveillance System:** [CDC Diabetes Atlas Surveillance](https://gis.cdc.gov/grasp/diabetes/diabetesatlas-surveillance.html)
- **Logistic regression concepts and implementation:** [Introduction to Statistical Learning](https://www.statlearning.com/)
- **Exploratory data analysis techniques:** [Hands-On Exploratory Data Analysis with Python](https://www.packtpub.com/product/hands-on-exploratory-data-analysis-with-python/9781800205549)
- **Model evaluation metrics:** [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- **Streamlit documentation for building web applications:** [Streamlit Documentation](https://docs.streamlit.io/)
- **Matplotlib and Plotly documentation for data visualization:** [Matplotlib Documentation](https://matplotlib.org/stable/contents.html), [Plotly Documentation](https://plotly.com/python/)
- **GitHub repository for sample code and project structure:** [Streamlit Cancer Prediction](https://github.com/alejandro-ao/streamlit-cancer-predict)
                          
## License
This project is licensed under the MIT License            

#### Project Repository
[GitHub Repository](https://github.com/priyam-choksi/Diabetes-Streamlit-App)

## Streamlit Application link

1. [Diabetes Prediction and Analysis Web Application](https://diabetes-prediction-and-analytics.streamlit.app/) - Comprehensive web application with Tabs for EDA and Prediction.
2. [Prediction Model](https://diabetes-pred-model.streamlit.app/) - Only Prediction Model
3. [Exploratory Data Analysis (EDA)](https://diabetes-eda.streamlit.app/) - Only EDA and Visualizations
      
    """)

#Prediction Model
# Function to load models
def load_models():
    model = joblib.load('diabetes_model.pkl')
    preprocessor = joblib.load('diabetes_preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_models()

# Main function for the app
def diabetes_app():
    st.title('🩺 Diabetes Prediction Tool')

    # Create a two-column layout with a smaller column for the sidebar
    sidebar_col, divider_col, content_col = st.columns([1.5 ,0.1, 4], gap="medium")

    with sidebar_col:
        # Sidebar with input sliders
        st.subheader("User Input")
        age = st.slider('Age', min_value=1, max_value=120, value=30, step=1)
        bmi = st.slider('BMI', min_value=10.0, max_value=50.0, value=22.0, step=0.1)
        glucose = st.slider('Glucose Level (mg/dL)', min_value=50, max_value=200, value=99, step=1)
        hbA1c = st.slider('HbA1c Level (%)', min_value=3.5, max_value=15.0, value=5.5, step=0.1)
        gender = st.radio('Gender', ['Female', 'Male'])

    with content_col:
        # Main content area
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
        type_2_diabetes_probability = diabetes_probability * 0.7
        feedback = provide_feedback(bmi, glucose, hbA1c)

        

        # Displaying the prediction results and health metrics in separate containers
        with st.container():
            st.subheader('👤 Your Health Metrics:')
            st.json({
                "Age": age,
                "BMI": f"{bmi} - {feedback['BMI']}",
                "Glucose Level": f"{glucose} mg/dL - {feedback['Glucose']}",
                "HbA1c Level": f"{hbA1c}% - {feedback['HbA1c']}"
            })

        with st.container():
            st.subheader('🔍 Diabetes Assessment')
            # Display each metric in its own row
            st.metric("Probability of Diabetes", f"{diabetes_probability*100:.2f}%")
            st.metric("Probability of Type 2 Diabetes", f"{type_2_diabetes_probability*100:.2f}%")
            status_color = "🔴" if diabetes_probability > 0.5 else "🟢"
            st.metric("Diagnosis", f"{status_color} {'Diabetic' if diabetes_probability > 0.5 else 'Not Diabetic'}")
            
    st.subheader('📈 Visual Health Analysis')
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


#EDA
data_url = "https://raw.githubusercontent.com/priyam-choksi/Diabetes-Streamlit-App/main/diabetes_dataset.csv"

@st.cache_resource
def load_data(data_url):
    """Loads data from a specified URL using pandas and caches it to avoid unnecessary reloads."""
    df = pd.read_csv(data_url)
    return df

def apply_filters(data, states, year_range, gender, age_range, hypertension, heart_disease, diabetes, smoking_status):
    """Applies filters to the data based on user input and returns the filtered data."""
    filtered_data = data.copy()
    if states:
        filtered_data = filtered_data[filtered_data['location'].isin(states)]
    filtered_data = filtered_data[(filtered_data['year'] >= year_range[0]) & (filtered_data['year'] <= year_range[1])]
    if gender != 'All':
        filtered_data = filtered_data[filtered_data['gender'] == gender]
    filtered_data = filtered_data[(filtered_data['age'] >= age_range[0]) & (filtered_data['age'] <= age_range[1])]
    if hypertension != 'All':
        filtered_data = filtered_data[filtered_data['hypertension'] == (hypertension == 'Yes')]
    if heart_disease != 'All':
        filtered_data = filtered_data[filtered_data['heart_disease'] == (heart_disease == 'Yes')]
    if diabetes != 'All':
        filtered_data = filtered_data[filtered_data['diabetes'] == (diabetes == 'Yes')]
    if smoking_status:
        filtered_data = filtered_data[filtered_data['smoking_history'].isin(smoking_status)]
    return filtered_data

def perform_eda():
    data_url = "https://raw.githubusercontent.com/priyam-choksi/Diabetes-Streamlit-App/main/diabetes_dataset.csv"
    data = load_data(data_url)
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()

    st.title('📊 Diabetes Data Interactive EDA')

    with st.expander("🗺️ State Filter"):
        selected_states = st.multiselect('Select State:', sorted(data['location'].unique()), default=sorted(data['location'].unique()))

    with st.expander("📅 Year Range Filter"):
        selected_year_range = st.slider('Select Year Range:', int(data['year'].min()), int(data['year'].max()), (int(data['year'].min()), int(data['year'].max())))

    with st.expander("👤 Gender Filter"):
        selected_gender = st.radio('Select Gender:', ['All'] + sorted(data['gender'].unique().tolist()))

    with st.expander("🔢 Age Range Filter"):
        selected_age_range = st.slider('Select Age Range:', int(data['age'].min()), int(data['age'].max()), (int(data['age'].min()), int(data['age'].max())))

    with st.expander("💗 Hypertension Filter"):
        selected_hypertension = st.radio('Hypertension:', ['All', 'Yes', 'No'])

    with st.expander("❤️ Heart Disease Filter"):
        selected_heart_disease = st.radio('Heart Disease:', ['All', 'Yes', 'No'])

    with st.expander("🩺 Diabetes Filter"):
        selected_diabetes = st.radio('Diabetes:', ['All', 'Yes', 'No'])

    with st.expander("🚬 Smoking Status Filter"):
        selected_smoking_status = st.multiselect('Smoking Status:', sorted(data['smoking_history'].unique()), default=sorted(data['smoking_history'].unique()))

    filtered_data = apply_filters(data, selected_states, selected_year_range, selected_gender, selected_age_range, selected_hypertension, selected_heart_disease, selected_diabetes, selected_smoking_status)

    
    st.header('Filtered Data:')
    st.dataframe(filtered_data, height=300)

    st.markdown("---")

    st.header('Data Visualization')

    st.markdown("---")

    st.header('Select Graphs')
    viz_type = st.selectbox('Select Visualization Type:', ['Histogram', 'Line Chart', 'Bar Chart', 'Radar Chart', 'Scatter Plot', 'Pair Plot', 'Box Plot', 'Correlation Matrix'])

    with st.expander("Feature Selection"):
        selected_features = st.multiselect('Select Features to Plot:', numeric_features, default=numeric_features)


    if viz_type == 'Histogram':
        for feature in selected_features:
            fig = px.histogram(filtered_data, x=feature, title=f'Histogram of {feature}', nbins=20)
            st.plotly_chart(fig)

    elif viz_type == 'Line Chart':
        fig, ax = plt.subplots()
        for feature in selected_features:
            ax.plot(filtered_data[feature], label=feature)
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

    elif viz_type == 'Bar Chart':
        for feature in selected_features:
            plt.figure(figsize=(10, 5))
            sns.barplot(x=filtered_data.index, y=filtered_data[feature])
            plt.title(f'Bar Chart of {feature}')
            plt.tight_layout()
            st.pyplot(plt)

    elif viz_type == 'Radar Chart':
        if len(selected_features) >= 3:
            radar_df = (filtered_data[selected_features] - filtered_data[selected_features].min()) / (filtered_data[selected_features].max() - filtered_data[selected_features].min())  # Normalizing data
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_df.iloc[0].values,
                theta=selected_features,
                fill='toself'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
            st.plotly_chart(fig)
        else:
            st.error('Please select at least three features for the radar chart.')

    elif viz_type == 'Scatter Plot':
        pair_features = list(combinations(selected_features, 2))  # All pairs of selected features
        for (x, y) in pair_features:
            plt.figure(figsize=(10, 5))
            sns.scatterplot(x=filtered_data[x], y=filtered_data[y])
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(f'Scatter Plot: {x} vs {y}')
            plt.tight_layout()
            st.pyplot(plt)

    elif viz_type == 'Pair Plot':
        if len(selected_features) > 1:
            pair_plot_fig = sns.pairplot(filtered_data[selected_features])
            st.pyplot(pair_plot_fig)
        else:
            st.error('Please select more than one feature for the pair plot.')

    elif viz_type == 'Box Plot':
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=filtered_data[selected_features])
        plt.title('Box Plot of Selected Features')
        plt.tight_layout()
        st.pyplot(plt)

    elif viz_type == 'Correlation Matrix':
        if len(selected_features) > 1:
            corr = filtered_data[selected_features].corr()
            fig = px.imshow(corr, text_auto=True, labels=dict(color='Correlation'))
            fig.update_layout(title='Correlation Matrix', xaxis_title='Features', yaxis_title='Features')
            st.plotly_chart(fig)
        else:
            st.error('Please select at least two features to display the correlation matrix.')


def main():
    tab1, tab2, tab3 = st.tabs(["Project Description", "Prediction Model", "EDA & Visualization"])
    
    with tab1:
        project_description()

    with tab2:
        diabetes_app()

    with tab3:
        perform_eda()

if __name__ == "__main__":
    main()
