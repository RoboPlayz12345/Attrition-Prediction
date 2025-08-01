import streamlit as st
import numpy as np
import pandas as pd
import joblib
# ================== Load Model and Threshold ==================
best_threshold = joblib.load("best_threshold.pkl")  # saved earlier

st.title("Employee Attrition Prediction App")

st.write("Fill in the employee details below to predict the likelihood of attrition.")

# ================== Define Inputs ==================
monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
age = st.number_input("Age", min_value=18, max_value=70, value=30)
overtime_yes = st.selectbox("OverTime", ["No", "Yes"])
daily_rate = st.number_input("Daily Rate", min_value=0, value=800)
total_working_years = st.number_input("Total Working Years", min_value=0, value=5)
monthly_rate = st.number_input("Monthly Rate", min_value=0, value=15000)
employee_number = st.number_input("Employee Number", min_value=1, value=1)
distance_from_home = st.number_input("Distance From Home (km)", min_value=0, value=5)
hourly_rate = st.number_input("Hourly Rate", min_value=0, value=30)
years_at_company = st.number_input("Years at Company", min_value=0, value=3)
satisfaction_score = st.slider("Satisfaction Score (0-10)", 0, 10, 7)
remote_stress_score = st.slider("Remote Stress Score (0-10)", 0, 10, 3)

# ================== Convert Input to DataFrame ==================
input_data = pd.DataFrame([[
    monthly_income, age, overtime_yes, daily_rate, total_working_years,
    monthly_rate, employee_number, distance_from_home, hourly_rate,
    years_at_company, satisfaction_score, remote_stress_score
]], columns=[
    'MonthlyIncome', 'Age', 'OverTime_Yes', 'DailyRate', 'TotalWorkingYears',
    'MonthlyRate', 'EmployeeNumber', 'DistanceFromHome', 'HourlyRate',
    'YearsAtCompany', 'SatisfactionScore', 'RemoteStressScore'
])

# ================== Predict ==================
if st.button("Predict Attrition"):
    proba = stack_model.predict_proba(input_data)[:, 1][0]
    prediction = int(proba >= best_threshold)

    st.write(f"**Probability of Attrition:** {proba:.2f}")
    if prediction == 1:
        st.error("⚠️ High risk of attrition!")
    else:
        st.success("✅ Low risk of attrition.")


