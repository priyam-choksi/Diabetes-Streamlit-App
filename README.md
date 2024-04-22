# Diabetes Prediction and Analysis

1. [Diabetes Prediction and Analysis Web Application](https://diabetes-prediction-and-analytics.streamlit.app/) - Comprehensive web application with Tabs for EDA and Prediction.
2. [Prediction Model](https://diabetes-pred-model.streamlit.app/) - Only Prediction Model
3. [Exploratory Data Analysis (EDA)](https://diabetes-eda.streamlit.app/) - Only EDA and Visualizations

## Project Video 

1. [Video](https://youtu.be/gIcCXWj7e-A) - A Video showcasing the project files and demo of the web application.

## Project Overview

This project aims to assess the likelihood of diabetes based on various health metrics provided by the user. The application leverages a Logistic Regression model, well-suited for binary classification tasks, to predict the onset of diabetes. It features a user-friendly web interface developed with Streamlit, enabling easy interaction and real-time prediction capabilities.

## Features

- **User-Friendly Interface**: Utilizes Streamlit to provide a simple, intuitive UI for inputting health metrics.
- **Immediate Predictions**: Delivers real-time diabetes predictions and Type 2 diabetes probabilities based on user inputs.
- **Visual Health Metrics**: Graphically displays user health metrics against standard ranges, offering immediate visual feedback.
- **Exploratory Data Analysis (EDA)**: Includes a section for statistical summaries and visualizations of the underlying data used for model training.

## How It Works

1. **Enter Health Information**: Input your health metrics using interactive sliders for Age, BMI, Glucose Level, and HbA1c Level.
2. **Receive Predictions**: The app calculates and displays your diabetes probability and where your health metrics stand in comparison to normal ranges.
3. **Explore Underlying Data**: Navigate to the EDA tab to view more data visualizations that provide insights into the dataset.

## Built With

- **Python**: Primary programming language.
- **Streamlit**: App framework for creating the web interface.
- **Pandas & Numpy**: For data manipulation and numerical calculations.
- **Matplotlib & Plotly**: For generating interactive visualizations.
- **Scikit-Learn**: Utilized for Logistic Regression model implementation.

## Model Information

- **Model Used**: Logistic Regression, which is effective for binary classification problems.
- **Performance**:
  - **Accuracy**: 0.9597 
  - **ROC AUC:** 0.9587
    
### Model Training Details

The Logistic Regression model was trained with the following considerations:
- **Max Iterations**: Set to 1000 to ensure convergence.
- **Features Used**: Includes age, BMI, glucose levels, HbA1c levels, gender, and smoking history.
- **Coefficient Analysis**: Provided to understand the influence of each feature on the model's predictions.

## Getting Started

### Prerequisites

- Python 3.8 or later.
- Installation of Python and necessary libraries.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Diabetes-Prediction-App.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Execute the following command in the terminal:
```bash
streamlit run streamlit-app.py
```

## Contributing

Contributions are welcome! Please feel free to fork the project, add your features, and submit a pull request.

## References
- CDC Diabetes Surveillance System: [CDC Diabetes Atlas Surveillance](https://gis.cdc.gov/grasp/diabetes/diabetesatlas-surveillance.html)
- Logistic regression concepts and implementation: [Introduction to Statistical Learning](https://www.statlearning.com/)
- Exploratory data analysis techniques: [Hands-On Exploratory Data Analysis with Python](https://www.packtpub.com/product/hands-on-exploratory-data-analysis-with-python/9781800205549)
- Model evaluation metrics: [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- Streamlit documentation for building web applications: [Streamlit Documentation](https://docs.streamlit.io/)
- Matplotlib and Plotly documentation for data visualization: [Matplotlib Documentation](https://matplotlib.org/stable/contents.html), [Plotly Documentation](https://plotly.com/python/)
- GitHub repository for sample code and project structure: [Streamlit Cancer Prediction](https://github.com/alejandro-ao/streamlit-cancer-predict)

## License
MIT License

Copyright (c) 2024 Priyam Deepak Choksi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contact

- **Priyam Deepak Choksi**: [choksi.pr@northeastern.edu](mailto:choksi.pr@northeastern.edu)
- **Project Link**: [[https://github.com/priyam-choksi/Diabetes-Streamlit-App](https://github.com/priyam-choksi/Diabetes-Streamlit-App)]

---
