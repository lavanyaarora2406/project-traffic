"""
Traffic Volume Prediction Using Machine Learning
================================================
This module builds ML models to predict traffic volume based on:
- Timestamps (hour, day, month, season)
- Weather conditions
- Holiday indicators
- Historical traffic patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_explore_data(filepath='traffic_data.csv'):
    """
    Load the traffic dataset and perform initial exploration.

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("=" * 60)
    print("LOADING AND EXPLORING DATA")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Display first few rows
    print("\n--- First 5 Rows ---")
    print(df.head())

    # Check data types and info
    print("\n--- Data Types ---")
    print(df.dtypes)

    # Check for missing values
    print("\n--- Missing Values ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values found")

    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"\nRemoved {initial_rows - len(df)} duplicate rows")
    print(f"Final dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Basic statistics
    print("\n--- Basic Statistics ---")
    print(df.describe())

    return df


def visualize_traffic_patterns(df):
    """
    Create visualizations to understand traffic patterns.
    """
    print("\n" + "=" * 60)
    print("VISUALIZING TRAFFIC PATTERNS")
    print("=" * 60)

    # Convert timestamp if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Traffic volume distribution
    sns.histplot(df['traffic_volume'], kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].set_title('Distribution of Traffic Volume')
    axes[0, 0].set_xlabel('Traffic Volume')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Traffic by hour of day
    if 'hour' in df.columns:
        hourly_traffic = df.groupby('hour')['traffic_volume'].mean()
        axes[0, 1].bar(hourly_traffic.index, hourly_traffic.values, color='green')
        axes[0, 1].set_title('Average Traffic Volume by Hour')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Average Traffic Volume')

    # 3. Traffic by day of week
    if 'day_of_week' in df.columns:
        daily_traffic = df.groupby('day_of_week')['traffic_volume'].mean()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        axes[0, 2].bar(range(len(daily_traffic)), daily_traffic.values, color='orange')
        axes[0, 2].set_title('Average Traffic Volume by Day of Week')
        axes[0, 2].set_xlabel('Day of Week')
        axes[0, 2].set_ylabel('Average Traffic Volume')
        axes[0, 2].set_xticks(range(len(daily_traffic)))
        axes[0, 2].set_xticklabels(days, rotation=45)

    # 4. Traffic by weather condition
    if 'weather_condition' in df.columns:
        weather_traffic = df.groupby('weather_condition')['traffic_volume'].mean().sort_values()
        axes[1, 0].barh(range(len(weather_traffic)), weather_traffic.values, color='purple')
        axes[1, 0].set_title('Average Traffic Volume by Weather Condition')
        axes[1, 0].set_xlabel('Average Traffic Volume')
        axes[1, 0].set_ylabel('Weather Condition')
        axes[1, 0].set_yticks(range(len(weather_traffic)))
        axes[1, 0].set_yticklabels(weather_traffic.index, rotation=0)

    # 5. Traffic by month
    if 'month' in df.columns:
        monthly_traffic = df.groupby('month')['traffic_volume'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1, 1].bar(monthly_traffic.index, monthly_traffic.values, color='red')
        axes[1, 1].set_title('Average Traffic Volume by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Average Traffic Volume')
        axes[1, 1].set_xticks(range(1, 13))
        axes[1, 1].set_xticklabels(months, rotation=45)

    # 6. Holiday vs Non-holiday traffic
    if 'is_holiday' in df.columns:
        holiday_traffic = df.groupby('is_holiday')['traffic_volume'].mean()
        axes[1, 2].bar(['Non-Holiday', 'Holiday'], holiday_traffic.values, color=['blue', 'gold'])
        axes[1, 2].set_title('Average Traffic: Holiday vs Non-Holiday')
        axes[1, 2].set_xlabel('Day Type')
        axes[1, 2].set_ylabel('Average Traffic Volume')

    plt.tight_layout()
    plt.savefig('traffic_patterns.png', dpi=300, bbox_inches='tight')
    print("\nVisualizations saved to 'traffic_patterns.png'")
    plt.show()

    # Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                    fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap of Numeric Features')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved to 'correlation_heatmap.png'")
        plt.show()


def preprocess_data(df):
    """
    Preprocess the data for machine learning models.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATA")
    print("=" * 60)

    # Make a copy
    data = df.copy()

    # Convert timestamp to datetime if it exists
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Extract time-based features if timestamp exists
    if 'timestamp' in data.columns:
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['day_of_month'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['year'] = data['timestamp'].dt.year
        data['season'] = ((data['month'] % 12 + 3) // 3).map({1: 'Winter', 2: 'Spring',
                                                               3: 'Summer', 4: 'Fall'})
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_rush_hour'] = ((data['hour'].isin([7, 8, 9, 16, 17, 18]))).astype(int)

    # Encode categorical weather conditions
    if 'weather_condition' in data.columns:
        le_weather = LabelEncoder()
        data['weather_encoded'] = le_weather.fit_transform(data['weather_condition'])
        print(f"\nWeather categories encoded: {list(le_weather.classes_)}")

    # Encode season if it exists (convert to numeric)
    if 'season' in data.columns:
        season_mapping = {'Winter': 1, 'Spring': 2, 'Summer': 3, 'Fall': 4}
        data['season'] = data['season'].map(season_mapping)

    # Ensure is_holiday exists
    if 'is_holiday' not in data.columns and 'holiday' in data.columns:
        data['is_holiday'] = (data['holiday'] == True).astype(int)

    # Select features for modeling
    feature_columns = []

    # Add time-based features
    for col in ['hour', 'day_of_week', 'day_of_month', 'month', 'season',
                'is_weekend', 'is_rush_hour', 'is_holiday']:
        if col in data.columns:
            feature_columns.append(col)

    # Add weather feature
    if 'weather_encoded' in data.columns:
        feature_columns.append('weather_encoded')

    # Add other numeric features that might be relevant
    for col in data.columns:
        if col not in feature_columns and col not in ['timestamp', 'weather_condition',
                                                       'holiday', 'traffic_volume', 'year']:
            if data[col].dtype in ['int64', 'float64']:
                feature_columns.append(col)

    print(f"\nSelected features: {feature_columns}")

    # Prepare X and y
    X = data[feature_columns]
    y = data['traffic_volume']

    # Handle any remaining missing values
    X = X.fillna(0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, feature_columns


def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models and compare their performance.

    Returns:
        dict: Dictionary of trained models and their metrics
    """
    print("\n" + "=" * 60)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 60)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regressor': DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'Random Forest Regressor': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }

        print(f"Train RMSE: {train_rmse:.2f}")
        print(f"Test RMSE:  {test_rmse:.2f}")
        print(f"Train MAE:  {train_mae:.2f}")
        print(f"Test MAE:   {test_mae:.2f}")
        print(f"Train R²:   {train_r2:.4f}")
        print(f"Test R²:    {test_r2:.4f}")

    return results


def compare_models(results):
    """
    Create visualizations to compare model performance.
    """
    print("\n" + "=" * 60)
    print("COMPARING MODEL PERFORMANCE")
    print("=" * 60)

    model_names = list(results.keys())

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. RMSE Comparison
    rmse_values = [results[name]['test_rmse'] for name in model_names]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = axes[0, 0].bar(model_names, rmse_values, color=colors)
    axes[0, 0].set_title('Test RMSE Comparison (Lower is Better)')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, rmse_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 2. MAE Comparison
    mae_values = [results[name]['test_mae'] for name in model_names]
    bars = axes[0, 1].bar(model_names, mae_values, color=colors)
    axes[0, 1].set_title('Test MAE Comparison (Lower is Better)')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, mae_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # 3. R² Score Comparison
    r2_values = [results[name]['test_r2'] for name in model_names]
    bars = axes[1, 0].bar(model_names, r2_values, color=colors)
    axes[1, 0].set_title('Test R² Score Comparison (Higher is Better)')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].tick_params(axis='x', rotation=15)
    axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Fit')
    for bar, val in zip(bars, r2_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    # 4. Train vs Test R² (Overfitting Check)
    train_r2 = [results[name]['train_r2'] for name in model_names]
    test_r2 = [results[name]['test_r2'] for name in model_names]
    x = np.arange(len(model_names))
    width = 0.35
    axes[1, 1].bar(x - width/2, train_r2, width, label='Train R²', color='#3498db')
    axes[1, 1].bar(x + width/2, test_r2, width, label='Test R²', color='#e74c3c')
    axes[1, 1].set_title('Train vs Test R² (Overfitting Check)')
    axes[1, 1].set_ylabel('R² Score')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=15)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison chart saved to 'model_comparison.png'")
    plt.show()

    # Actual vs Predicted plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, name in enumerate(model_names):
        y_test = results[name]['y_pred_test']
        actual_values = list(results.values())[idx]['y_pred_test']
        # Get actual y_test values from the first model result
        if idx == 0:
            y_test_actual = list(results.values())[0]['y_pred_test']

        axes[idx].scatter(list(range(len(y_test))), sorted(y_test),
                         alpha=0.3, label='Predicted', color='red')
        axes[idx].scatter(list(range(len(y_test))), sorted(list(results.values())[0]['y_pred_test']),
                         alpha=0.3, label='Actual', color='blue')
        axes[idx].set_title(f'{name}\nPredictions')
        axes[idx].set_xlabel('Sample Index')
        axes[idx].set_ylabel('Traffic Volume')
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
    print("Predictions plot saved to 'predictions_plot.png'")
    plt.show()

    # Print summary table
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Model':<30} {'RMSE':<12} {'MAE':<12} {'R² Score':<12}")
    print("-" * 80)
    for name in model_names:
        r = results[name]
        print(f"{name:<30} {r['test_rmse']:<12.2f} {r['test_mae']:<12.2f} {r['test_r2']:<12.4f}")
    print("=" * 80)

    # Find best model
    best_model = min(results.keys(), key=lambda x: results[x]['test_rmse'])
    print(f"\nBest Model (by RMSE): {best_model}")

    return results


def save_models(results, feature_names):
    """
    Save trained models and preprocessing objects for later use.
    """
    print("\n" + "=" * 60)
    print("SAVING MODELS")
    print("=" * 60)

    # Save each model
    for name, data in results.items():
        model_filename = f"{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(data['model'], model_filename)
        print(f"Saved: {model_filename}")

    # Save feature names
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Saved: feature_names.pkl")

    # Save all results summary
    summary = {}
    for name, data in results.items():
        summary[name] = {
            'test_rmse': data['test_rmse'],
            'test_mae': data['test_mae'],
            'test_r2': data['test_r2']
        }
    joblib.dump(summary, 'model_summary.pkl')
    print("Saved: model_summary.pkl")

    print("\nAll models and artifacts saved successfully!")


def predict_traffic(model_path, feature_values, feature_names):
    """
    Make predictions using a saved model.

    Args:
        model_path: Path to saved model
        feature_values: Dictionary of feature names and values
        feature_names: List of feature names used during training

    Returns:
        float: Predicted traffic volume
    """
    model = joblib.load(model_path)
    features = joblib.load(feature_names)

    # Create feature vector in correct order
    X = np.array([[feature_values.get(f, 0) for f in features]])

    prediction = model.predict(X)[0]
    return prediction


def main():
    """
    Main function to run the complete traffic prediction pipeline.
    """
    print("\n" + "=" * 60)
    print("TRAFFIC VOLUME PREDICTION USING MACHINE LEARNING")
    print("=" * 60)

    # Step 1: Load and explore data
    df = load_and_explore_data()

    # Step 2: Visualize patterns
    visualize_traffic_patterns(df)

    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # Step 4: Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Step 5: Compare models
    compare_models(results)

    # Step 6: Save models
    save_models(results, feature_names)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - traffic_patterns.png (data visualizations)")
    print("  - correlation_heatmap.png (feature correlations)")
    print("  - model_comparison.png (model performance comparison)")
    print("  - predictions_plot.png (prediction visualizations)")
    print("  - *_model.pkl (trained models)")
    print("  - feature_names.pkl (feature list)")
    print("  - model_summary.pkl (performance summary)")


if __name__ == "__main__":
    main()
