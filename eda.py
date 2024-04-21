import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from itertools import combinations


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

    with st.expander("Feature Selection"):
        selected_features = st.multiselect('Select Features to Plot:', numeric_features, default=numeric_features)

    st.header('Visualizations')
    viz_type = st.selectbox('Select Visualization Type:', ['Histogram', 'Line Chart', 'Bar Chart', 'Radar Chart', 'Scatter Plot', 'Pair Plot', 'Box Plot', 'Correlation Matrix'])

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

if __name__ == "__main__":
    perform_eda()
