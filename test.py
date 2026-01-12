"""
Vehicle Emission Prediction - Industrial-Level Interactive Dashboard
Advanced Analytics Platform with Real-time Insights
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import time
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Vehicle Emission Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION: LOCAL FILE PATHS ---
FILE_PATHS = {
    'model': 'model.h5',
    'scaler': 'scaler.pkl',
    'encoders': 'label_encoders.pkl',
    'metadata': 'metadata.pkl'
}

# --- FUEL MAPPING (MALAYSIA CONTEXT) ---
FUEL_MAP = {
    'RON 95 (Regular Petrol)': 'X',
    'RON 97 (Premium Petrol)': 'Z',
    'RON 100 (High Performance)': 'Z',
    'Diesel (Euro 5 / Euro 2M)': 'D',
    'Ethanol (E85)': 'E',
    'Natural Gas (NGV)': 'N'
}

# Enhanced CSS
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    h1 {color: #1a237e; font-weight: 700; border-bottom: 3px solid #667eea;}
    h2 {color: #283593;}
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.75rem;
        font-weight: 600; border-radius: 8px;
        transition: all 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {transform: translateY(-2px);}
    [data-testid="stMetricValue"] {font-size: 2rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
for key in ['model', 'scaler', 'label_encoders', 'metadata', 'history', 'load_time', 'auto_load_attempted']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'history' and key != 'auto_load_attempted' else ([] if key == 'history' else False)

# --- AUTO-LOAD LOGIC ---
def load_local_artifacts():
    try:
        if os.path.exists(FILE_PATHS['model']) and \
           os.path.exists(FILE_PATHS['scaler']) and \
           os.path.exists(FILE_PATHS['encoders']):

            st.session_state.model = load_model(FILE_PATHS['model'])
            with open(FILE_PATHS['scaler'], 'rb') as f:
                st.session_state.scaler = pickle.load(f)
            with open(FILE_PATHS['encoders'], 'rb') as f:
                st.session_state.label_encoders = pickle.load(f)
            if os.path.exists(FILE_PATHS['metadata']):
                with open(FILE_PATHS['metadata'], 'rb') as f:
                    st.session_state.metadata = pickle.load(f)
            st.session_state.load_time = datetime.now()
            return True, "Success"
        else:
            return False, "Files not found"
    except Exception as e:
        return False, str(e)

if not st.session_state.auto_load_attempted and not st.session_state.model:
    success, msg = load_local_artifacts()
    st.session_state.auto_load_attempted = True
    if success:
        st.toast("✅ Local model files loaded automatically!")

# Sidebar
with st.sidebar:
    st.markdown("# 🌍 AI Dashboard")
    st.markdown("### Emission Analytics")
    st.markdown("---")
    st.markdown("## 📦 Model Manager")

    if st.session_state.model:
        st.success("**🟢 SYSTEM ONLINE**")
        if st.button("🔄 Reload / Reset", use_container_width=True):
            for key in ['model', 'scaler', 'label_encoders', 'metadata']:
                st.session_state[key] = None
            st.session_state.auto_load_attempted = False
            st.rerun()
    else:
        st.warning("**🔴 SYSTEM OFFLINE**")
        model_file = st.file_uploader("Model (.h5)", type=['h5'])
        scaler_file = st.file_uploader("Scaler (.pkl)", type=['pkl'])
        encoders_file = st.file_uploader("Encoders (.pkl)", type=['pkl'])
        metadata_file = st.file_uploader("Metadata (.pkl)", type=['pkl'])

        if st.button("🔄 Manual Load", use_container_width=True):
            if all([model_file, scaler_file, encoders_file]):
                try:
                    with open('temp.h5', 'wb') as f:
                        f.write(model_file.read())
                    st.session_state.model = load_model('temp.h5')
                    st.session_state.scaler = pickle.load(scaler_file)
                    st.session_state.label_encoders = pickle.load(encoders_file)
                    if metadata_file:
                        st.session_state.metadata = pickle.load(metadata_file)
                    st.session_state.load_time = datetime.now()
                    st.success("✅ Loaded!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# Header
st.title("🌍 Vehicle Emission Analytics Platform")
st.markdown("### AI-Powered Classification System")

# Main content
if not st.session_state.model:
    st.info("👋 Please upload your model files or ensure they are in the directory to start.")
else:
    tabs = st.tabs(["🔮 Predict", "📊 Batch", "📈 Analytics", "💡 Insights"])

    # TAB 1: SINGLE PREDICTION
    with tabs[0]:
        st.markdown("## 🔮 Single Vehicle Prediction")

        with st.form("pred_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**🔧 Engine Config**")
                engine = st.slider("Engine Size (L)", 0.5, 10.0, 2.5, 0.1)
                cylinders = st.select_slider("Cylinders", options=[3, 4, 5, 6, 8, 10, 12, 16], value=4)

                trans_opts = sorted(st.session_state.label_encoders.get('Transmission', {}).classes_.tolist()) if 'Transmission' in st.session_state.label_encoders else ['A6', 'A8', 'AV', 'CVT']
                transmission = st.selectbox("Transmission", trans_opts)

            with c2:
                st.markdown("**⛽ Fuel Consumption**")
                city = st.slider("City (L/100km)", 1.0, 30.0, 10.0, 0.1)
                hwy = st.slider("Highway (L/100km)", 1.0, 25.0, 7.0, 0.1)

                comb = city * 0.55 + hwy * 0.45
                mpg = int(235.214 / comb) if comb > 0 else 0
                st.info(f"Combined: **{comb:.2f}** L/100km  \nMPG: **{mpg}**")

            with c3:
                st.markdown("**🚙 Vehicle Details**")
                make_opts = sorted(st.session_state.label_encoders.get('Make', {}).classes_.tolist()) if 'Make' in st.session_state.label_encoders else ['Toyota', 'Honda']
                make = st.selectbox("Make", make_opts)

                model_opts = sorted(st.session_state.label_encoders.get('Model', {}).classes_.tolist()) if 'Model' in st.session_state.label_encoders else ['Accord', 'Camry']
                model = st.selectbox("Model", model_opts)

                class_opts = sorted(st.session_state.label_encoders.get('Vehicle Class', {}).classes_.tolist()) if 'Vehicle Class' in st.session_state.label_encoders else ['Compact', 'Mid-size']
                vclass = st.selectbox("Class", class_opts)

                ftype_label = st.selectbox("Fuel Type", list(FUEL_MAP.keys()))

            submit = st.form_submit_button("🎯 PREDICT", use_container_width=True, type="primary")

        if submit:
            try:
                ftype_code = FUEL_MAP[ftype_label]

                # Prepare data
                data = pd.DataFrame({
                    'Engine Size(L)': [engine],
                    'Cylinders': [cylinders],
                    'Fuel Consumption City (L/100 km)': [city],
                    'Fuel Consumption Hwy (L/100 km)': [hwy],
                    'Fuel Consumption Comb (L/100 km)': [comb],
                    'Fuel Consumption Comb (mpg)': [mpg]
                })

                # Encode categorical
                for col, val in [('Make', make), ('Model', model), ('Vehicle Class', vclass),
                                 ('Transmission', transmission)]:
                    if col in st.session_state.label_encoders:
                        try:
                            data[f'{col}_encoded'] = st.session_state.label_encoders[col].transform([val])[0]
                        except:
                            data[f'{col}_encoded'] = 0
                    else:
                        data[f'{col}_encoded'] = 0

                # Handle Fuel Type separately
                if 'Fuel Type' in st.session_state.label_encoders:
                    try:
                        data['Fuel Type_encoded'] = st.session_state.label_encoders['Fuel Type'].transform([ftype_code])[0]
                    except:
                        data['Fuel Type_encoded'] = 0

                # Predict
                scaled = st.session_state.scaler.transform(data)
                proba = st.session_state.model.predict(scaled, verbose=0)[0][0]

                # --- CHANGED: Renamed to Risk Index ---
                risk_index = proba * 100

                # Determine Status
                if risk_index <= 40:
                    status = "LOW EMISSION"
                    sub_status = "Likely Eco-Friendly"
                    color_theme = "linear-gradient(135deg,#51cf66,#43a047)" # Green
                    gauge_color = "#51cf66"
                    icon = "🍃"
                elif risk_index <= 60:
                    status = "MODERATE RISK"
                    sub_status = "Borderline / Uncertain"
                    color_theme = "linear-gradient(135deg,#fcc419,#f08c00)" # Orange
                    gauge_color = "#fcc419"
                    icon = "⚠️"
                else:
                    status = "HIGH EMISSION"
                    sub_status = "Likely High Polluter"
                    color_theme = "linear-gradient(135deg,#ff6b6b,#e03131)" # Red
                    gauge_color = "#ff6b6b"
                    icon = "🏭"

                # Save history
                st.session_state.history.append({
                    'time': datetime.now(),
                    'make': make,
                    'model': model,
                    'status': status,
                    'risk_index': risk_index # Saved as risk_index now
                })

                st.success("✅ Prediction Complete!")

                # Result card
                st.markdown("---")
                _, c, _ = st.columns([1, 2, 1])
                with c:
                    st.markdown(f"""
                    <div style='text-align:center; padding:2rem; background:{color_theme}; border-radius:15px; color:white; box-shadow: 0 10px 15px rgba(0,0,0,0.1);'>
                        <h1 style='color:white; margin:0; font-size:4rem;'>{icon}</h1>
                        <h2 style='color:white; margin-top:10px;'>{status}</h2>
                        <p style='font-size:1.5rem; opacity:0.9'>{sub_status}</p>
                        <hr style='border-color:rgba(255,255,255,0.3); margin:15px 0;'>
                        <p style='font-size:1.2rem; margin-bottom:0;'>Risk Index</p>
                        <p style='font-size:2.5rem; font-weight:bold; margin-top:0;'>{risk_index:.0f} / 100</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Gauge (Risk Index)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_index,
                    title={'text': "Emission Risk Index", 'font': {'size': 20}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 40], 'color': '#d4edda'},   # Safe Zone
                            {'range': [40, 60], 'color': '#fff3cd'},  # Uncertain Zone
                            {'range': [60, 100], 'color': '#f8d7da'}  # Danger Zone
                        ],
                        'threshold': {'line': {'color': 'black', 'width': 4}, 'value': risk_index}
                    }
                ))
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")

    # TAB 2: BATCH
    with tabs[1]:
        st.markdown("## 📊 Batch Classification")

        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            df = pd.read_csv(uploaded)
            if st.button("🚀 Process Batch"):
                results = []
                prog = st.progress(0)

                for idx, row in df.iterrows():
                    prog.progress((idx + 1) / len(df))
                    try:
                        results.append({'Make': row.get('Make'), 'Model': row.get('Model'), 'Status': 'Processed'})
                    except:
                        pass

                prog.empty()
                st.dataframe(pd.DataFrame(results))

    # TAB 3: ANALYTICS
    with tabs[2]:
        st.markdown("## 📈 Classification History")

        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)

            # Key check for safety
            if 'status' not in hist_df.columns:
                st.warning("⚠️ Old history format detected. Some charts may be hidden.")
                hist_df['status'] = 'Unknown'

            c1, c2, c3 = st.columns(3)
            c1.metric("Low Predictions", len(hist_df[hist_df['status']=='LOW EMISSION']))
            c2.metric("Moderate/Uncertain", len(hist_df[hist_df['status']=='MODERATE RISK']))
            c3.metric("High Predictions", len(hist_df[hist_df['status']=='HIGH EMISSION']))

            if not hist_df[hist_df['status'] != 'Unknown'].empty:
                fig = px.pie(hist_df, names='status', title="Prediction Distribution",
                             color='status',
                             color_discrete_map={
                                 'LOW EMISSION':'#51cf66',
                                 'MODERATE RISK':'#fcc419',
                                 'HIGH EMISSION':'#ff6b6b',
                                 'Unknown': '#e0e0e0'
                             })
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions yet. Make a prediction in the 'Predict' tab!")

    # TAB 4: INSIGHTS
    with tabs[3]:
        st.markdown("## 💡 Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.info("""
            **Understanding the Risk Index:**
            - **0 - 40:** **Safe Zone.** Strong indicators of efficiency.
            - **40 - 60:** **Watch List.** Borderline performance.
            - **60 - 100:** **Action Required.** Strong indicators of high emissions.
            """)
        with c2:
            st.warning("""
            **Fuel Type Guide:**
            - **RON 95:** Standard 'X' Grade (Regular)
            - **RON 97/100:** Premium 'Z' Grade (High Octane)
            - **Diesel:** Standard 'D' Grade
            """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center; color:#666;'>Vehicle Emission Analytics Platform</div>", unsafe_allow_html=True)
