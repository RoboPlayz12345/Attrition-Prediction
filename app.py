import streamlit as st
import pandas as pd
import joblib

# ===== Load the single PKL file =====
model_data = joblib.load("final_stacked_model.pkl")
stack_model = model_data["model"]
best_threshold = model_data["threshold"]

st.title("Employee Attrition Prediction App")
st.write("Fill in employee details to predict the attrition risk.")

# ===== User Inputs =====
monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
age = st.number_input("Age", min_value=18, max_value=70, value=30)
overtime_yes = st.selectbox("OverTime", ["No", "Yes"])
total_working_years = st.number_input("Total Working Years", min_value=0, value=5)
distance_from_home = st.number_input("Distance From Home (km)", min_value=0, value=5)
years_at_company = st.number_input("Years at Company", min_value=0, value=3)
satisfaction_score = st.slider("Satisfaction Score (0-10)", 0, 10, 7)
remote_stress_score = st.slider("Remote Stress Score (0-10)", 0, 10, 3)

# Convert OverTime to numeric 
overtime_val = 1 if overtime_yes == "Yes" else 0

# ===== Prepare Input DataFrame =====
input_data = pd.DataFrame([[
    monthly_income, age, overtime_val, total_working_years,
    distance_from_home, years_at_company,
    satisfaction_score, remote_stress_score
]], columns=[
    'MonthlyIncome', 'Age', 'OverTime_Yes', 'TotalWorkingYears',
    'DistanceFromHome', 'YearsAtCompany',
    'SatisfactionScore', 'RemoteStressScore'
])

# ===== Predict on Button Click =====
if st.button("Predict Attrition"):
    proba = stack_model.predict_proba(input_data)[:, 1][0]
    prediction = int(proba >= best_threshold)

    
    if prediction == 1:
        st.error("⚠️ High risk of attrition!")
    else:
        st.success("✅ Low risk of attrition.")






