import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Job Salary Predictor",  # This sets the browser tab title
    page_icon="💼",                     # Optional: Add a nice emoji or icon
    layout="centered"
)

# Load saved model, scaler, and label encoders
model = joblib.load("xgb_salary_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Mappings for readable options
experience_mapping = {
    "Entry-level (EN)": "EN",
    "Mid-level (MI)": "MI",
    "Senior-level (SE)": "SE",
    "Executive (EX)": "EX"
}

employment_mapping = {
    "Part-time (PT)": "PT",
    "Full-time (FT)": "FT",
    "Contract (CT)": "CT",
    "Freelance (FL)": "FL"
}

company_size_mapping = {
    "Small (S)": "S",
    "Medium (M)": "M",
    "Large (L)": "L"
}

# Function to get readable options
def get_label_options(encoder):
    return list(encoder.classes_)

st.title("💼 Job Salary Predictor")

st.markdown("Fill in the job details to predict the **estimated salary (in USD)**")

# Original label categories from dataset
job_title = st.selectbox("Job Title", get_label_options(label_encoders['job_title']))
company_location = st.selectbox("Company Location", get_label_options(label_encoders['company_location']))
employee_residence = st.selectbox("Employee Residence", get_label_options(label_encoders['employee_residence']))

# User-friendly dropdowns
experience_level_readable = st.selectbox("Experience Level", list(experience_mapping.keys()))
employment_type_readable = st.selectbox("Employment Type", list(employment_mapping.keys()))
company_size_readable = st.selectbox("Company Size", list(company_size_mapping.keys()))

remote_ratio = st.slider("Remote Work Ratio (%)", 0, 100, 50)

# Convert readable values to original codes
experience_level = experience_mapping[experience_level_readable]
employment_type = employment_mapping[employment_type_readable]
company_size = company_size_mapping[company_size_readable]

# Encode using saved encoders
def encode_input(feature, value):
    return label_encoders[feature].transform([value])[0]

job_title_encoded = encode_input('job_title', job_title)
company_location_encoded = encode_input('company_location', company_location)
employee_residence_encoded = encode_input('employee_residence', employee_residence)
experience_level_encoded = encode_input('experience_level', experience_level)
employment_type_encoded = encode_input('employment_type', employment_type)
company_size_encoded = encode_input('company_size', company_size)

# Combine all input values
input_data = np.array([[job_title_encoded, company_location_encoded, employee_residence_encoded,
                        experience_level_encoded, employment_type_encoded, company_size_encoded,
                        remote_ratio]])

# Apply scaler
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("🔮 Predict Salary"):
    salary = model.predict(input_scaled)[0]
    st.success(f"💵 Estimated Salary: **${salary:,.2f} USD**")
