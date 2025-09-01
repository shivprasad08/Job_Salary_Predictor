import streamlit as st
import joblib
import pandas as pd
import numpy as np
import logging
import time

# --- Page Configuration ---
# This MUST be the first Streamlit command.
st.set_page_config(
    page_title="AI Job Salary Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching Data and Models ---
@st.cache_resource
def load_model_and_components():
    """Loads and caches the ML model, scaler, and encoders."""
    try:
        model = joblib.load('xgb_salary_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        logging.info("Model and components loaded successfully from disk.")
        return model, scaler, label_encoders
    except FileNotFoundError as e:
        logging.error(f"Error loading model components: {e}")
        st.error("A required model file was not found. Please ensure all .pkl files are present.")
        return None, None, None

@st.cache_data
def load_data_for_ui():
    """Loads and caches the dataset for UI Adropdowns."""
    try:
        df = pd.read_csv('ai_job_dataset.csv')
        logging.info("Dataset for UI loaded successfully.")
        return df
    except FileNotFoundError:
        logging.error("Dataset file 'ai_job_dataset.csv' not found.")
        st.error("The 'ai_job_dataset.csv' file is missing.")
        return None

# --- Load all necessary resources ---
model, scaler, label_encoders = load_model_and_components()
df = load_data_for_ui()

# --- Main UI Layout ---
st.title("🤖 AI Job Salary Predictor")
st.markdown("""
Welcome to the AI Job Salary Predictor. This tool leverages a machine learning model 
to estimate salaries based on job characteristics. The underlying XGBoost Regressor model 
was trained on a comprehensive dataset of AI job postings.
""")
st.info("The model's performance on test data achieved an **R² score of 0.881**.", icon="💡")

# --- Sidebar for User Inputs ---
st.sidebar.header("Enter Job Details")
st.sidebar.markdown("Use the form below to get a salary prediction.")

if df is not None:
    job_title = st.sidebar.selectbox("Job Title", sorted(df['job_title'].unique()))
    experience_level = st.sidebar.selectbox("Experience Level", ['EN', 'MI', 'SE', 'EX'], help="EN: Entry-level, MI: Mid-level, SE: Senior-level, EX: Executive-level")
    employment_type = st.sidebar.selectbox("Employment Type", sorted(df['employment_type'].unique()))
    company_location = st.sidebar.selectbox("Company Location", sorted(df['company_location'].unique()))
    employee_residence = st.sidebar.selectbox("Employee Residence", sorted(df['employee_residence'].unique()))
    company_size = st.sidebar.selectbox("Company Size", ['S', 'M', 'L'], help="S: Small, M: Medium, L: Large")
    remote_ratio = st.sidebar.slider("Remote Work Ratio (%)", 0, 100, 50)
else:
    st.sidebar.error("Input fields cannot be displayed as the required dataset is missing.")

# --- Prediction Logic and Display ---
if 'predicted_salary' not in st.session_state:
    st.session_state.predicted_salary = None

if st.sidebar.button("Predict Salary", type="primary"):
    if all([model, scaler, label_encoders, df is not None]):
        with st.spinner('Analyzing...'):
            time.sleep(1) 
            try:
                # Create a DataFrame from user inputs
                input_data = pd.DataFrame({
                    'job_title': [job_title],
                    'company_location': [company_location],
                    'employee_residence': [employee_residence],
                    'experience_level': [experience_level],
                    'employment_type': [employment_type],
                    'company_size': [company_size],
                    'remote_ratio': [remote_ratio]
                })

                # Preprocessing Pipeline
                encoded_data = input_data.copy()
                for col, le in label_encoders.items():
                    if col in encoded_data.columns:
                        encoded_data[col] = encoded_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                
                scaled_data = scaler.transform(encoded_data)
                
                # Make Prediction
                prediction = model.predict(scaled_data)
                st.session_state.predicted_salary = prediction[0]
                
                logging.info(f"Prediction successful: ${st.session_state.predicted_salary:,.2f}")

            except Exception as e:
                logging.error(f"An error occurred during prediction: {e}")
                st.error("An unexpected error occurred. Please try again.")
    else:
        st.error("The application is not ready for predictions. Missing model components.")

if st.session_state.predicted_salary is not None:
    st.subheader("Prediction Result")
    st.success(f"**Estimated Annual Salary: ${st.session_state.predicted_salary:,.2f} USD**")

