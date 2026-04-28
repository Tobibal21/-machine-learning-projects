import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Power Demand Generator",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Power Demand Simulation App")
st.markdown("""
This app simulates hourly power demand data for 30 days, including:
- Daily and weekly patterns
- Temperature effects
- Industrial demand influence
- Random noise
""")

# Sidebar controls
st.sidebar.header("Simulation Parameters")

# Parameters for simulation
col1, col2 = st.sidebar.columns(2)
with col1:
    days = st.slider("Number of days", 7, 90, 30)
    base_demand = st.slider("Base Demand (MW)", 30, 100, 50)
    
with col2:
    temp_effect = st.slider("Temperature Effect", 1, 5, 2)
    noise_level = st.slider("Noise Level (MW)", 1, 10, 3)

# Advanced parameters
with st.sidebar.expander("Advanced Parameters"):
    daily_amplitude = st.slider("Daily Cycle Amplitude", 5, 20, 10)
    weekday_boost = st.slider("Weekday Boost (MW)", 0, 15, 5)
    industrial_range = st.slider("Industrial Index Range", 0.5, 2.0, (0.5, 1.5))

if st.sidebar.button("🔄 Generate New Dataset", type="primary"):
    st.cache_data.clear()

# Main simulation function
@st.cache_data
def generate_power_demand(days, base_demand, temp_effect, noise_level, 
                         daily_amplitude, weekday_boost, industrial_range):
    np.random.seed(42)
    
    hours = 24 * days
    time = np.arange(hours)
    
    # Features
    hour_of_day = time % 24
    day_of_week = (time // 24) % 7
    
    temperature = 25 + 5 * np.sin(2 * np.pi * hour_of_day / 24) + np.random.normal(0, 1, hours)
    industrial_index = np.random.uniform(industrial_range[0], industrial_range[1], hours)
    
    # Base demand formula
    demand = (
        base_demand
        + daily_amplitude * np.sin(2 * np.pi * hour_of_day / 24)
        + weekday_boost * (day_of_week < 5)
        + temp_effect * temperature
        + 20 * industrial_index
        + np.random.normal(0, noise_level, hours)
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        "hour": hour_of_day,
        "day_of_week": day_of_week,
        "temperature": temperature,
        "industrial_index": industrial_index,
        "demand": demand
    })
    
    # Add lag features
    df["lag_1"] = df["demand"].shift(1)
    df["lag_24"] = df["demand"].shift(24)
    
    # Add datetime for better visualization
    start_date = datetime(2024, 1, 1)
    df["datetime"] = [start_date + timedelta(hours=i) for i in range(hours)]
    df["day_name"] = df["datetime"].dt.day_name()
    
    df = df.dropna()
    
    return df

# Generate data
df = generate_power_demand(days, base_demand, temp_effect, noise_level,
                          daily_amplitude, weekday_boost, industrial_range)

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Visualizations", "📥 Download Data", "📐 Feature Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Date Range", f"{df['datetime'].min().date()} to {df['datetime'].max().date()}")
        
    with col2:
        st.metric("Avg Demand (MW)", f"{df['demand'].mean():.1f}")
        st.metric("Peak Demand (MW)", f"{df['demand'].max():.1f}")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("Data Statistics")
    st.dataframe(df[['temperature', 'industrial_index', 'demand', 'lag_1', 'lag_24']].describe(), 
                use_container_width=True)

with tab2:
    # Demand over time
    st.subheader("Power Demand Over Time")
    fig1 = px.line(df, x="datetime", y="demand", 
                   title="Hourly Power Demand",
                   labels={"demand": "Demand (MW)", "datetime": "Date"})
    fig1.update_layout(hovermode='x unified')
    st.plotly_chart(fig1, use_container_width=True)
    
    # Two columns for smaller plots
    col1, col2 = st.columns(2)
    
    with col1:
        # Demand by hour of day
        st.subheader("Demand by Hour of Day")
        hourly_pattern = df.groupby("hour")["demand"].agg(["mean", "std"]).reset_index()
        fig2 = px.line(hourly_pattern, x="hour", y="mean", 
                       error_y="std",
                       title="Average Daily Pattern",
                       labels={"hour": "Hour of Day", "mean": "Avg Demand (MW)"})
        fig2.add_vline(x=9, line_dash="dash", line_color="green", annotation_text="Morning Peak")
        fig2.add_vline(x=18, line_dash="dash", line_color="orange", annotation_text="Evening Peak")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Demand by day of week
        st.subheader("Demand by Day of Week")
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_pattern = df.groupby("day_name")["demand"].mean().reindex(dow_order).reset_index()
        fig3 = px.bar(weekday_pattern, x="day_name", y="demand",
                      title="Average Demand by Day",
                      labels={"day_name": "Day", "demand": "Avg Demand (MW)"},
                      color="demand", color_continuous_scale="Viridis")
        st.plotly_chart(fig3, use_container_width=True)
    
    # Scatter plots
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Demand vs Temperature")
        fig4 = px.scatter(df, x="temperature", y="demand", 
                         title="Temperature Impact on Demand",
                         labels={"temperature": "Temperature (°C)", "demand": "Demand (MW)"},
                         opacity=0.6, trendline="ols")
        st.plotly_chart(fig4, use_container_width=True)
    
    with col4:
        st.subheader("Demand vs Industrial Index")
        fig5 = px.scatter(df, x="industrial_index", y="demand",
                         title="Industrial Impact on Demand",
                         labels={"industrial_index": "Industrial Index", "demand": "Demand (MW)"},
                         opacity=0.6, trendline="ols")
        st.plotly_chart(fig5, use_container_width=True)
    
    # 3D plot
    st.subheader("3D View: Temperature, Industrial Index, and Demand")
    fig6 = px.scatter_3d(df.sample(min(1000, len(df))), 
                         x='temperature', y='industrial_index', z='demand',
                         color='hour', title="3D Relationship",
                         labels={'temperature': 'Temperature (°C)', 
                                'industrial_index': 'Industrial Index',
                                'demand': 'Demand (MW)'})
    st.plotly_chart(fig6, use_container_width=True)

with tab3:
    st.subheader("Download Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file_format = st.radio("Choose file format:", ["CSV", "Parquet", "Excel"])
    
    with col2:
        include_lags = st.checkbox("Include lag features", value=True)
    
    if not include_lags:
        download_df = df[['datetime', 'hour', 'day_of_week', 'day_name', 'temperature', 'industrial_index', 'demand']]
    else:
        download_df = df
    
    # Create download buttons
    if file_format == "CSV":
        csv = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"power_demand_{days}days.csv",
            mime="text/csv",
            use_container_width=True
        )
    elif file_format == "Parquet":
        with st.spinner("Preparing Parquet file..."):
            parquet = download_df.to_parquet(index=False)
            st.download_button(
                label="📥 Download Parquet",
                data=parquet,
                file_name=f"power_demand_{days}days.parquet",
                mime="application/octet-stream",
                use_container_width=True
            )
    else:  # Excel
        with st.spinner("Preparing Excel file..."):
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                download_df.to_excel(writer, sheet_name='Power Demand', index=False)
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"power_demand_{days}days.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Show dataset info
    st.subheader("Dataset Information")
    st.info(f"""
    **Dataset Stats:**
    - Rows: {len(download_df):,}
    - Columns: {len(download_df.columns)}
    - Features: {', '.join(download_df.columns)}
    - Missing values: {download_df.isnull().sum().sum()}
    """)

with tab4:
    st.subheader("Feature Correlation Analysis")
    
    # Correlation matrix
    corr_cols = ['temperature', 'industrial_index', 'demand', 'lag_1', 'lag_24']
    corr_matrix = df[corr_cols].corr()
    
    fig7 = px.imshow(corr_matrix, 
                     text_auto=True, 
                     aspect="auto",
                     color_continuous_scale="RdBu_r",
                     title="Feature Correlation Matrix")
    st.plotly_chart(fig7, use_container_width=True)
    
    # Feature importance analysis
    st.subheader("Feature Impact on Demand")
    
    # Real vs simulated components
    components = pd.DataFrame({
        'Component': ['Base', 'Daily Cycle', 'Weekday Effect', 'Temperature', 'Industrial', 'Noise'],
        'Contribution (MW)': [
            base_demand,
            daily_amplitude,
            weekday_boost,
            temp_effect * df['temperature'].mean(),
            20 * df['industrial_index'].mean(),
            noise_level
        ]
    })
    
    fig8 = px.bar(components, x='Component', y='Contribution (MW)',
                  title="Average Contribution to Power Demand",
                  color='Contribution (MW)',
                  color_continuous_scale='Viridis')
    st.plotly_chart(fig8, use_container_width=True)
    
    # Time series decomposition
    st.subheader("Time Series Components")
    
    # Calculate components separately
    df_analysis = df.copy()
    df_analysis['daily_cycle'] = daily_amplitude * np.sin(2 * np.pi * df_analysis['hour'] / 24)
    df_analysis['weekday_effect'] = weekday_boost * (df_analysis['day_of_week'] < 5)
    df_analysis['temp_contribution'] = temp_effect * df_analysis['temperature']
    df_analysis['industrial_contribution'] = 20 * df_analysis['industrial_index']
    
    # Sample a week for clarity
    week_data = df_analysis.head(168)  # 7 days
    fig9 = go.Figure()
    fig9.add_trace(go.Scatter(x=week_data['datetime'], y=week_data['demand'], 
                              name='Total Demand', line=dict(color='blue', width=2)))
    fig9.add_trace(go.Scatter(x=week_data['datetime'], y=week_data['daily_cycle'], 
                              name='Daily Cycle', line=dict(color='green', dash='dot')))
    fig9.add_trace(go.Scatter(x=week_data['datetime'], y=week_data['weekday_effect'], 
                              name='Weekday Effect', line=dict(color='orange', dash='dot')))
    fig9.update_layout(title="Demand Components (First Week)",
                       xaxis_title="Date",
                       yaxis_title="Demand (MW)",
                       hovermode='x unified')
    st.plotly_chart(fig9, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("⚡ **Power Demand Simulation App** | Created with Streamlit")