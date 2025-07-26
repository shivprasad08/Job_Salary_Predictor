import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv("ai_job_dataset.csv")  # update path if needed

# Remove duplicates and missing target values
df.drop_duplicates(inplace=True)
df.rename(columns={'salary_usd': 'salary_in_usd'}, inplace=True)
df.dropna(subset=['salary_in_usd'], inplace=True)

# Fill missing values in categorical columns with mode
cat_cols = ['job_title', 'company_location', 'employee_residence',
            'experience_level', 'employment_type', 'company_size']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Initialize label encoders
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features & target
X = df[['job_title', 'company_location', 'employee_residence',
        'experience_level', 'employment_type', 'company_size', 'remote_ratio']]
y = df['salary_in_usd']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = XGBRegressor()
model.fit(X_scaled, y)

# Save model, scaler, encoders
joblib.dump(model, "xgb_salary_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ Model, Scaler, and Encoders saved successfully.")
