import streamlit as st
from predictionapp import diabetes_app  # Correct import assuming trying.py is in the same directory
from eda import perform_eda  # Correct import assuming EDA.py is setup and in the same directory

@st.cache_data
def project_description():
    st.title("Diabetes Prediction and Analysis Web Application")
    st.write("""
---
#### Project Overview
This Diabetes Prediction App aims to assess the likelihood of diabetes based on various health metrics provided by the user. The application leverages a Logistic Regression model, well-suited for binary classification tasks, to predict the onset of diabetes. It features a user-friendly web interface developed with Streamlit, enabling easy interaction and real-time prediction capabilities.
The application is divided into three main components:

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
  - **Accuracy**: Achieved on the test set.
  - **ROC AUC**: Measure of the model's ability to distinguish between the classes.
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
      
    """)
    
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