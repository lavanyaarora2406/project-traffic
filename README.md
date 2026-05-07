# Traffic Volume Prediction Using Machine Learning

A comprehensive machine learning project that predicts traffic volume on roadways using historical data and environmental factors.

## Project Overview

This project builds a complete ML pipeline to forecast traffic congestion patterns. It demonstrates:
- Data cleaning and exploratory data analysis (EDA)
- Feature engineering from timestamps
- Multiple regression model training and comparison
- Model evaluation and persistence

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Sample Data (Optional)

If you don't have real traffic data, generate synthetic data:

```bash
python generate_sample_data.py
```

### Step 2: Run the Main Pipeline

```bash
python traffic_prediction.py
```

This will:
1. Load and explore the dataset
2. Generate visualizations of traffic patterns
3. Preprocess data and engineer features
4. Train three regression models
5. Compare model performance
6. Save trained models for deployment

## Project Structure

```
project-traffic/
├── traffic_prediction.py      # Main ML pipeline
├── generate_sample_data.py    # Synthetic data generator
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── traffic_data.csv           # Input dataset (generated or real)
```

## Output Files

After running the pipeline, you'll get:

| File | Description |
|------|-------------|
| `traffic_patterns.png` | Visualizations of traffic patterns |
| `correlation_heatmap.png` | Feature correlation matrix |
| `model_comparison.png` | Model performance comparison charts |
| `predictions_plot.png` | Actual vs predicted values |
| `linear_regression_model.pkl` | Trained Linear Regression model |
| `decision_tree_regressor_model.pkl` | Trained Decision Tree model |
| `random_forest_regressor_model.pkl` | Trained Random Forest model |
| `feature_names.pkl` | Feature list for predictions |
| `model_summary.pkl` | Performance metrics summary |

## Models

Three regression models are trained and compared:

1. **Linear Regression** - Baseline model for linear relationships
2. **Decision Tree Regressor** - Captures non-linear patterns
3. **Random Forest Regressor** - Ensemble model for improved accuracy

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error) - Penalizes large errors
- **MAE** (Mean Absolute Error) - Average prediction error
- **R² Score** - Variance explained by the model

## Using Saved Models

```python
import joblib
import numpy as np

# Load model and feature names
model = joblib.load('random_forest_regressor_model.pkl')
features = joblib.load('feature_names.pkl')

# Create feature vector (order matters!)
# Example: [hour, day_of_week, day_of_month, month, is_weekend, is_rush_hour, is_holiday, weather_encoded]
X = np.array([[8, 1, 15, 4, 0, 1, 0, 2]])  # Rush hour on a weekday

# Predict
prediction = model.predict(X)[0]
print(f"Predicted traffic volume: {prediction}")
```

## Features

| Feature | Type | Description |
|---------|------|-------------|
| hour | Numeric | Hour of day (0-23) |
| day_of_week | Numeric | Day of week (0=Monday, 6=Sunday) |
| day_of_month | Numeric | Day of month (1-31) |
| month | Numeric | Month (1-12) |
| is_weekend | Binary | 1 if Saturday/Sunday |
| is_rush_hour | Binary | 1 during peak hours (7-9, 16-18) |
| is_holiday | Binary | 1 if holiday |
| weather_encoded | Numeric | Encoded weather condition |

## Customization

### For Real Data

Replace `traffic_data.csv` with your real traffic sensor data. Ensure it has these columns:
- `timestamp` - Date/time of observation
- `traffic_volume` - Target variable (vehicle count)
- `weather_condition` - Categorical weather description
- `is_holiday` - Binary holiday indicator

### Hyperparameter Tuning

Modify model parameters in `train_models()`:

```python
RandomForestRegressor(
    n_estimators=200,      # More trees
    max_depth=20,          # Deeper trees
    min_samples_split=2    # More splits
)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

## License

MIT License
