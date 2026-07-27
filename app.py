import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import time

# Page configuration
st.set_page_config(page_title="WorkTrack Predictor", layout="centered")

st.title("WorkTrack Predictor")
st.write("Drag the time slider to see how the Stacking Ensemble model predicts the log type in real-time.")

# Load the trained Stacking model
@st.cache_resource
def load_model():
    return joblib.load('stacking_model.pkl')

try:
    model = load_model()
    
    st.header("Time Simulator")
    
    # HH:MM slider restricted from 08:00 to 19:00
    user_time = st.slider(
        "Select simulation time:",
        min_value=time(8, 0),
        max_value=time(19, 0),
        value=time(8, 0),
        format="HH:mm"
    )

    # Convert HH:MM back into seconds_from_midnight for X
    seconds_from_midnight = (user_time.hour * 3600) + (user_time.minute * 60)
    st.info(f"Feature value sent to the AI: {seconds_from_midnight} seconds elapsed since midnight.")

    # Create the input DataFrame for X
    input_data = pd.DataFrame([[seconds_from_midnight]], columns=['seconds_from_midnight'])

    # Real-time prediction (updates instantly when dragging the slider)
    raw_prediction = model.predict(input_data)[0] # Extract the numeric prediction (0, 1, 2, or 3)
    probabilities = model.predict_proba(input_data)[0]
    
    # Mapping dictionary to translate numbers into readable labels
    label_map = {
        0: "BREAK END",
        1: "BREAK START",
        2: "CLOCK IN",
        3: "CLOCK OUT"
    }
    
    # Convert the numerical prediction to string
    readable_prediction = label_map.get(int(raw_prediction), f"UNKNOWN (Class {raw_prediction})")
    
    # Map classes for the probability labels
    readable_classes = [label_map.get(int(c), f"Class {c}") for c in model.classes_]
    
    # Display the final prediction output
    st.success(f"Predicted Log Type: {readable_prediction}")
    
       # 4. Progress Bars showing model confidence (Replacing the old bar chart)
    st.write("Stacking Ensemble Confidence Level:")
    
    # Create a clean dictionary from the arrays for direct lookup
    prob_dict = {label_map.get(int(c), f"Class {c}"): p for c, p in zip(model.classes_, probabilities)}
    
    # Define a fixed order for the rows so they don't jump around when sliding
    fixed_order = ["CLOCK IN", "BREAK START", "BREAK END", "CLOCK OUT"]
    
    for action in fixed_order:
        # Get the probability for the current action (default to 0.0 if not found)
        prob_val = prob_dict.get(action, 0.0)
        percentage = round(prob_val * 100, 1)
        
        # Display the text and the numerical percentage in a single clean row
        st.write(f"**{action}** ({percentage}%)")
        
        # Display the horizontal progress bar (st.progress accepts values from 0.0 to 1.0)
        st.progress(float(prob_val))

except FileNotFoundError:
    st.error("File 'stacking_model.pkl' not found in the repository.")
