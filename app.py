import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Traffic Volume Prediction Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    .status-low {
        color: #2ecc71;
        font-weight: bold;
    }
    .status-moderate {
        color: #f39c12;
        font-weight: bold;
    }
    .status-high {
        color: #e67e22;
        font-weight: bold;
    }
    .status-congested {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load models and configs
@st.cache_resource
def load_ml_resources():
    models = {
        'Random Forest Regressor': joblib.load('random_forest_regressor_model.pkl'),
        'Decision Tree Regressor': joblib.load('decision_tree_regressor_model.pkl'),
        'Linear Regression': joblib.load('linear_regression_model.pkl')
    }
    feature_names = joblib.load('feature_names.pkl')
    try:
        model_summary = joblib.load('model_summary.pkl')
    except Exception:
        model_summary = None
    return models, feature_names, model_summary

# Load models and features
try:
    models, feature_names, model_summary = load_ml_resources()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or feature names. Please ensure model training was completed and pickle files are present. Error: {e}")
    models_loaded = False

# App title and header
st.title("🚗 Intelligent Traffic Volume Prediction Dashboard")
st.markdown("Use this interactive dashboard to predict traffic volume based on date, time, weather, and environmental factors using pre-trained machine learning models.")

if models_loaded:
    # Weather categories
    weather_categories = ['Clear', 'Cloudy', 'Fog', 'Heavy Rain', 'Light Rain', 'Snow']
    weather_mapping = {cat: idx for idx, cat in enumerate(sorted(weather_categories))}
    # Alphabetical order: Clear=0, Cloudy=1, Fog=2, Heavy Rain=3, Light Rain=4, Snow=5

    # Season mapping
    # 1: Winter, 2: Spring, 3: Summer, 4: Fall
    
    # Sidebar layout for user inputs
    st.sidebar.header("🕹️ Prediction Parameters")
    
    # Model selection
    selected_model_name = st.sidebar.selectbox(
        "Select Machine Learning Model",
        list(models.keys()),
        index=0  # Default to Random Forest Regressor
    )
    
    st.sidebar.markdown("---")
    
    # Date and Time inputs
    selected_date = st.sidebar.date_input("Select Date", datetime.today())
    selected_hour = st.sidebar.slider("Select Hour of Day", 0, 23, 12, format="%02d:00")
    
    # Weather selection
    selected_weather = st.sidebar.selectbox("Weather Condition", weather_categories, index=0)
    
    # Holiday selection
    is_holiday_input = st.sidebar.checkbox("Is it a Holiday?", value=False)
    
    # Extract features from inputs
    hour = selected_hour
    day_of_week = selected_date.weekday() # 0 = Monday, 6 = Sunday
    day_of_month = selected_date.day
    month = selected_date.month
    
    # Season extraction: Winter=1, Spring=2, Summer=3, Fall=4
    season = ((month % 12 + 3) // 3)
    
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    is_holiday = 1 if is_holiday_input else 0
    weather_encoded = weather_mapping[selected_weather]
    
    # Construct feature mapping dictionary
    feature_values = {
        'hour': hour,
        'day_of_week': day_of_week,
        'day_of_month': day_of_month,
        'month': month,
        'season': season,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour,
        'is_holiday': is_holiday,
        'weather_encoded': weather_encoded
    }
    
    # Display features debug info (collapsed)
    with st.sidebar.expander("🛠️ Engineered Features (Debug)"):
        st.write(feature_values)

    # Main dashboard layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Real-time Traffic Prediction")
        
        # Predict using selected model
        model = models[selected_model_name]
        
        # Ensure correct order of features
        X_pred = np.array([[feature_values[f] for f in feature_names]])
        prediction = model.predict(X_pred)[0]
        prediction_val = int(max(0, prediction))
        
        # Determine traffic congestion density level
        if prediction_val < 1500:
            traffic_density = "Low Traffic"
            status_class = "status-low"
            density_desc = "Smooth traffic conditions. Normal speeds and minimal delays expected."
            color_hex = "#2ecc71"
        elif prediction_val < 3000:
            traffic_density = "Moderate Traffic"
            status_class = "status-moderate"
            density_desc = "Typical traffic volume. Normal speed limits with occasional slowdowns at key junctions."
            color_hex = "#f39c12"
        elif prediction_val < 4500:
            traffic_density = "High Traffic"
            status_class = "status-high"
            density_desc = "Heavy congestion. Traffic speeds are noticeably reduced. Expect minor to moderate delays."
            color_hex = "#e67e22"
        else:
            traffic_density = "Congested"
            status_class = "status-congested"
            density_desc = "Gridlock or stop-and-go conditions. Heavy delays. Recommend alternative routes or travel times."
            color_hex = "#e74c3c"
            
        # Metric card layout
        st.markdown(f"""
        <div class="metric-card">
            <h4>PREDICTED VEHICLE VOLUME</h4>
            <h1 style="color: {color_hex}; font-size: 3.5rem; margin: 10px 0;">{prediction_val:,} <span style="font-size: 1.2rem; color: #7f8c8d;">vehicles / hour</span></h1>
            <p style="font-size: 1.2rem;">Traffic Class: <span class="{status_class}">{traffic_density}</span></p>
            <p style="color: #7f8c8d; font-size: 0.95rem; margin-top: 5px;">{density_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Compare prediction across all models
        st.subheader("📊 Model Comparison")
        comp_data = []
        for name, m in models.items():
            pred = int(max(0, m.predict(X_pred)[0]))
            comp_data.append({"Model Name": name, "Predicted Volume (veh/hr)": pred})
        
        st.table(pd.DataFrame(comp_data))

    with col2:
        st.subheader("📈 24-Hour Forecast Profile")
        st.markdown(f"Hourly volume profile for **{selected_date.strftime('%B %d, %Y')}** ({selected_date.strftime('%A')})")
        
        # Calculate prediction for all 24 hours
        hours_list = list(range(24))
        forecast_predictions = []
        
        for h in hours_list:
            h_is_rush = 1 if h in [7, 8, 9, 16, 17, 18] else 0
            h_features = {
                'hour': h,
                'day_of_week': day_of_week,
                'day_of_month': day_of_month,
                'month': month,
                'season': season,
                'is_weekend': is_weekend,
                'is_rush_hour': h_is_rush,
                'is_holiday': is_holiday,
                'weather_encoded': weather_encoded
            }
            X_forecast = np.array([[h_features[f] for f in feature_names]])
            pred_h = max(0, model.predict(X_forecast)[0])
            forecast_predictions.append(int(pred_h))
            
        # Draw line chart using streamlit native chart
        chart_data = pd.DataFrame({
            'Hour of Day': hours_list,
            'Predicted Volume': forecast_predictions
        }).set_index('Hour of Day')
        
        st.line_chart(chart_data)
        
        # Quick summary indicators
        min_vol_hour = hours_list[np.argmin(forecast_predictions)]
        max_vol_hour = hours_list[np.argmax(forecast_predictions)]
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Peak Hour of Day", f"{max_vol_hour:02d}:00", f"{max(forecast_predictions):,} vehicles", delta_color="inverse")
        with col_s2:
            st.metric("Lowest Volume Hour", f"{min_vol_hour:02d}:00", f"{min(forecast_predictions):,} vehicles")

    # Tabs for analytics and model info
    st.markdown("---")
    st.subheader("🔍 Model Training Insights & Historical Analysis")
    
    tab_summary, tab_patterns, tab_correlation, tab_comparison = st.tabs([
        "📋 Model Evaluation Summary",
        "🗺️ Historical Patterns (EDA)",
        "🌡️ Feature Correlation Matrix",
        "🎯 Model vs Actual Fit"
    ])
    
    with tab_summary:
        st.markdown("### Model Performance Metrics")
        st.markdown("These metrics were generated during model evaluation on the test set:")
        
        if model_summary:
            metrics_df = pd.DataFrame(model_summary).T
            # Rename columns for clarity
            metrics_df.columns = ['RMSE (Root Mean Sq. Error)', 'MAE (Mean Absolute Error)', 'R² Score (Variance Explained)']
            st.dataframe(metrics_df.style.format({'RMSE (Root Mean Sq. Error)': '{:.2f}', 'MAE (Mean Absolute Error)': '{:.2f}', 'R² Score (Variance Explained)': '{:.4f}'}))
            
            st.markdown("""
            - **RMSE (Root Mean Squared Error)**: Standard deviation of residuals. Lower is better.
            - **MAE (Mean Absolute Error)**: Average absolute prediction error. Lower is better.
            - **R² Score**: Proportion of variance in the traffic volume that is predictable from features. Closer to 1.0 is better.
            """)
        else:
            st.warning("Model summary dictionary could not be loaded.")
            
    with tab_patterns:
        st.markdown("### Historical Traffic Patterns")
        st.markdown("Visualizations of average traffic trends by hour, day of the week, weather, and holidays:")
        if os.path.exists("traffic_patterns.png"):
            st.image("traffic_patterns.png")
        else:
            st.info("The file 'traffic_patterns.png' was not found. Run 'traffic_prediction.py' to generate it.")
            
    with tab_correlation:
        st.markdown("### Feature Correlation Heatmap")
        st.markdown("Understanding how different factors correlate with each other and target volume:")
        if os.path.exists("correlation_heatmap.png"):
            st.image("correlation_heatmap.png")
        else:
            st.info("The file 'correlation_heatmap.png' was not found. Run 'traffic_prediction.py' to generate it.")
            
    with tab_comparison:
        st.markdown("### Predictions Comparison & Model Performance Chart")
        
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("#### Performance Metrics Visualized")
            if os.path.exists("model_comparison.png"):
                st.image("model_comparison.png")
            else:
                st.info("The file 'model_comparison.png' was not found.")
        with col_i2:
            st.markdown("#### Actual vs Predicted Trend Line")
            if os.path.exists("predictions_plot.png"):
                st.image("predictions_plot.png")
            else:
                st.info("The file 'predictions_plot.png' was not found.")

else:
    st.warning("Please verify that you have generated data and trained the machine learning models before running the app. You can do this by running:")
    st.code("python generate_sample_data.py\npython traffic_prediction.py", language="bash")
