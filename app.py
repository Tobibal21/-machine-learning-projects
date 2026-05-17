import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Machine Learning Projects", page_icon="🤖", layout="wide")

st.title("🤖 My Machine Learning Projects Showcase")
st.markdown("Welcome! This app hosts interactive versions of the machine learning models I've built.")

tab1, tab2 = st.tabs(["Experience-Driven Salary Estimation", "More Coming Soon!"])

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
    st.header("Stay Tuned! 🚀")
    st.markdown("I am continuously adding more interactive models from my Jupyter Notebooks. Check back soon for Air Pollution and Power Consumption models!")
