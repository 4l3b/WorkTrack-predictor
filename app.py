import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page configuration
st.set_page_config(page_title="WorkTrack Predictor", layout="centered")

st.title("WorkTrack Predictor - Stacking Ensemble Showcase")
st.write("Drag the time slider to see how the Stacking Ensemble model predicts the log type in real-time.")

# 1. Load the trained Stacking model
@st.cache_resource
def load_model():
    return joblib.load('stacking_model.pkl')

try:
    model = load_model()
    
    st.header("24-Hour Time Simulator")
    
    # Continuous slider
    seconds_from_midnight = st.slider(
        "Select seconds elapsed:", 
        min_value=28800, 
        max_value=68400, 
        value=28800,
        step=60
    )

    # Convert seconds HH:MM format
    hours = seconds_from_midnight // 3600
    minutes = (seconds_from_midnight % 3600) // 60
    st.info(f"Equivalent Time: {hours:02d}:{minutes:02d} | Feature value: {seconds_from_midnight} seconds.")

    # Create the input DataFrame for X
    input_data = pd.DataFrame([[seconds_from_midnight]], columns=['seconds_from_midnight'])

    # Real-time prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[0]
    
    # Display the final prediction output
    st.success(f"Predicted Log Type: {str(prediction[0]).upper()}")
    
    # Bar chart showing model confidence
    st.write("Stacking Ensemble Confidence Level:")
    
    prob_df = pd.DataFrame({
        'Log Action': model.classes_,
        'Probability (%)': [round(p * 100, 2) for p in probabilities]
    }).sort_values(by='Probability (%)', ascending=False)
    
    st.bar_chart(data=prob_df, x='Log Action', y='Probability (%)', use_container_width=True)

except FileNotFoundError:
    st.error("File 'stacking_model.pkl' not found in the project repository.")
