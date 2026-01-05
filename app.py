import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# ------------------------------
# Load the trained model, scaler, and label encoders
# ------------------------------
model = load_model('vehicle_emission_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="Vehicle Emission Predictor",
    layout="wide"
)

# Custom CSS for bigger fonts
st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; }
    .medium-font { font-size:24px !important; }
    .small-font { font-size:18px !important; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# App Title
# ------------------------------
st.markdown('<p class="big-font">🚗 Vehicle Emission Level Predictor 🌱</p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">Predict whether a vehicle has low or high CO₂ emissions based on its specifications.</p>', unsafe_allow_html=True)

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Enter Vehicle Details:")

def user_input():
    # Numerical features
    engine_size = st.sidebar.slider("Engine Size (L)", 1.0, 6.5, 2.0, 0.1)
    cylinders = st.sidebar.selectbox("Cylinders", [3,4,5,6,8,10,12,16])
    fuel_city = st.sidebar.slider("Fuel Consumption City (L/100 km)", 4.0, 25.0, 8.0, 0.1)
    fuel_hwy = st.sidebar.slider("Fuel Consumption Hwy (L/100 km)", 3.0, 18.0, 6.0, 0.1)
    fuel_comb_l = st.sidebar.slider("Fuel Consumption Comb (L/100 km)", 4.0, 20.0, 7.0, 0.1)
    fuel_comb_mpg = st.sidebar.slider("Fuel Consumption Comb (mpg)", 10.0, 60.0, 35.0, 0.1)

    # Categorical features
    make = st.sidebar.selectbox("Make", label_encoders['Make'].classes_)
    model_name = st.sidebar.selectbox("Model", label_encoders['Model'].classes_)
    vehicle_class = st.sidebar.selectbox("Vehicle Class", label_encoders['Vehicle Class'].classes_)
    transmission = st.sidebar.selectbox("Transmission", label_encoders['Transmission'].classes_)
    fuel_type = st.sidebar.selectbox("Fuel Type", label_encoders['Fuel Type'].classes_)

    data = {
        'Engine Size(L)': engine_size,
        'Cylinders': cylinders,
        'Fuel Consumption City (L/100 km)': fuel_city,
        'Fuel Consumption Hwy (L/100 km)': fuel_hwy,
        'Fuel Consumption Comb (L/100 km)': fuel_comb_l,
        'Fuel Consumption Comb (mpg)': fuel_comb_mpg,
        'Make_encoded': label_encoders['Make'].transform([make])[0],
        'Model_encoded': label_encoders['Model'].transform([model_name])[0],
        'Vehicle Class_encoded': label_encoders['Vehicle Class'].transform([vehicle_class])[0],
        'Transmission_encoded': label_encoders['Transmission'].transform([transmission])[0],
        'Fuel Type_encoded': label_encoders['Fuel Type'].transform([fuel_type])[0],
    }
    return pd.DataFrame([data])

input_df = user_input()

# ------------------------------
# Display input
# ------------------------------
st.subheader("🚘 Input Vehicle Features")
st.dataframe(input_df.style.set_properties(**{'font-size':'18px'}))

# ------------------------------
# Scale input
# ------------------------------
input_scaled = scaler.transform(input_df)

# ------------------------------
# Predict
# ------------------------------
prediction_prob = float(model.predict(input_scaled)[0][0])
prediction_class = "High Emission" if prediction_prob > 0.5 else "Low Emission"

st.subheader("🎯 Prediction Result")
st.markdown(f'<p class="big-font">Predicted Emission Level: {prediction_class}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="medium-font">Probability of High Emission: {prediction_prob:.2f}</p>', unsafe_allow_html=True)

# ------------------------------
# Probability Gauge
# ------------------------------
st.subheader("📊 Probability Gauge")
st.progress(int(prediction_prob * 100))

# ------------------------------
# Optional Tip
# ------------------------------
if prediction_prob > 0.5:
    st.markdown('<p class="small-font">💡 Tip: Consider a smaller engine or more efficient fuel type to reduce emissions.</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="small-font">✅ This vehicle is relatively eco-friendly.</p>', unsafe_allow_html=True)
