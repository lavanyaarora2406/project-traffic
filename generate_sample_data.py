"""
Generate Sample Traffic Data
============================
Creates realistic synthetic traffic data for demonstration purposes.
In production, replace this with real traffic sensor data.
"""



import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_traffic_data(n_rows=10000, output_file='traffic_data.csv'):
    """
    Generate synthetic traffic data with realistic patterns.

    Parameters:
        n_rows: Number of rows to generate
        output_file: Output CSV filename
    """
    np.random.seed(42)
    random.seed(42)

    print(f"Generating {n_rows} rows of synthetic traffic data...")

    # Generate timestamps (2 years of hourly data)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(n_rows)]

    # Extract time components
    hours = [ts.hour for ts in timestamps]
    days_of_week = [ts.weekday() for ts in timestamps]
    months = [ts.month for ts in timestamps]

    # Weather conditions with realistic distribution
    weather_options = ['Clear', 'Cloudy', 'Light Rain', 'Heavy Rain', 'Snow', 'Fog']
    weather_weights = [0.45, 0.25, 0.15, 0.08, 0.04, 0.03]
    weather_conditions = random.choices(weather_options, weights=weather_weights, k=n_rows)

    # Holiday indicator (roughly 10% of days)
    is_holiday = np.random.choice([0, 1], size=n_rows, p=[0.9, 0.1])

    # Generate traffic volume with realistic patterns
    traffic_volume = []

    for i in range(n_rows):
        hour = hours[i]
        dow = days_of_week[i]
        month = months[i]
        weather = weather_conditions[i]
        holiday = is_holiday[i]

        # Base traffic volume
        base = 3000

        # Hour effect (rush hours have higher traffic)
        if hour in [7, 8, 9]:  # Morning rush
            hour_factor = 1.8
        elif hour in [16, 17, 18]:  # Evening rush
            hour_factor = 2.0
        elif hour in [10, 11, 12, 13, 14, 15]:  # Midday
            hour_factor = 1.3
        elif hour in [19, 20, 21]:  # Evening
            hour_factor = 1.1
        elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Night
            hour_factor = 0.4
        else:
            hour_factor = 1.0

        # Day of week effect (weekends have different patterns)
        if dow >= 5:  # Weekend
            dow_factor = 0.7
            # Rush hours are less pronounced on weekends
            if hour in [10, 11, 12, 13, 14, 15, 16]:
                hour_factor = 1.4
        else:  # Weekday
            dow_factor = 1.0

        # Month/season effect
        if month in [6, 7, 8]:  # Summer
            month_factor = 1.1
        elif month in [12, 1, 2]:  # Winter
            month_factor = 0.9
        else:
            month_factor = 1.0

        # Weather effect
        weather_effects = {
            'Clear': 1.0,
            'Cloudy': 0.95,
            'Light Rain': 0.85,
            'Heavy Rain': 0.7,
            'Snow': 0.6,
            'Fog': 0.75
        }
        weather_factor = weather_effects[weather]

        # Holiday effect
        holiday_factor = 0.5 if holiday == 1 else 1.0

        # Calculate traffic volume
        volume = base * hour_factor * dow_factor * month_factor * weather_factor * holiday_factor

        # Add random noise
        noise = np.random.normal(0, volume * 0.15)
        volume = max(0, volume + noise)

        traffic_volume.append(int(volume))

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'hour': hours,
        'day_of_week': days_of_week,
        'month': months,
        'weather_condition': weather_conditions,
        'is_holiday': is_holiday,
        'traffic_volume': traffic_volume
    })

    # Add some duplicate rows (to demonstrate deduplication)
    duplicate_indices = np.random.choice(n_rows, size=int(n_rows * 0.02), replace=True)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data saved to '{output_file}'")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    return df


if __name__ == "__main__":
    df = generate_traffic_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
