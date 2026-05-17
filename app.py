import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Machine Learning Projects", page_icon="🤖", layout="wide")

st.title("🤖 My Machine Learning Projects Showcase")
st.markdown("Welcome! This app hosts interactive versions of the machine learning models I've built.")

tab1, tab2, tab3 = st.tabs(["Experience-Driven Salary Estimation", "Air Pollution vs Industrial Activity", "Power Consumption Simulator"])

with tab1:
    st.header("💼 Experience-Driven Salary Estimation")
    st.markdown("This tool uses a Linear Regression model to predict salary based on years of experience.")
    
    # Load Data
    @st.cache_data
    def load_data():
        return pd.read_csv("Experience-Driven Salary Estimation/Salary_Data.csv")
    
    df = load_data()
    
    # Train Model
    X = df[['YearsExperience']]
    y = df['Salary']
    model = LinearRegression()
    model.fit(X, y)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Predict Your Salary")
        exp_input = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        
        if st.button("Predict 🚀"):
            prediction = model.predict([[exp_input]])[0]
            st.success(f"**Predicted Salary:** ${prediction:,.2f}")
            
        st.markdown("---")
        st.write("Model Equation:")
        st.code(f"Salary = {model.coef_[0]:.2f} * Experience + {model.intercept_:.2f}")

    with col2:
        st.subheader("Data & Model Fit")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(X, y, color='blue', label='Actual Data')
        
        # Plot regression line
        x_range = np.linspace(0, df['YearsExperience'].max() + 5, 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        ax.plot(x_range, y_pred, color='red', linewidth=2, label='Regression Line')
        
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary ($)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

with tab2:
    st.header("🏭 Air Pollution vs Industrial Activity")
    st.markdown("This tool uses a Multiple Linear Regression model to predict PM2.5 pollution levels based on industrial output, traffic density, and temperature.")
    
    @st.cache_data
    def load_pollution_data():
        return pd.read_csv("Air pollution vs industrial activity/pollution_data.csv")
    
    df_poll = load_pollution_data()
    
    X_poll = df_poll[['industrial_output', 'traffic_density', 'temperature']]
    y_poll = df_poll['pm25']
    poll_model = LinearRegression()
    poll_model.fit(X_poll, y_poll)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Predict Pollution Level")
        ind_input = st.number_input("Industrial Output (Index):", min_value=0, max_value=100, value=30, step=5)
        traf_input = st.number_input("Traffic Density (Vehicles/hr):", min_value=0, max_value=100, value=40, step=5)
        temp_input = st.number_input("Temperature (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=1.0)
        
        if st.button("Predict PM2.5 🌫️"):
            pred_pm25 = poll_model.predict([[ind_input, traf_input, temp_input]])[0]
            st.warning(f"**Predicted PM2.5 Level:** {pred_pm25:.1f} µg/m³")
            
    with col2:
        st.subheader("Data Overview")
        st.dataframe(df_poll)
        st.write("Model Coefficients:")
        st.write(f"- Industrial Output: {poll_model.coef_[0]:.2f}")
        st.write(f"- Traffic Density: {poll_model.coef_[1]:.2f}")
        st.write(f"- Temperature: {poll_model.coef_[2]:.2f}")

with tab3:
    st.header("⚡ Power Consumption Simulator")
    st.info("I noticed you already built a comprehensive 300+ line Streamlit app for Power Consumption! You can deploy it directly by setting your Main file path to `power consumption/app.py` in Streamlit Cloud.")
