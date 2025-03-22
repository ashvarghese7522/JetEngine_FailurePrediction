import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    """Load cleaned dataset"""
    df = pd.read_csv(file_path)
    return df


def add_rolling_features(df, window_size=5):
    """Add rolling mean and standard deviation for each sensor feature"""
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    for col in sensor_cols:
        df[f"{col}_rolling_mean"] = df.groupby("unit_number")[col].rolling(window=window_size,
                                                                           min_periods=1).mean().reset_index(0,
                                                                                                             drop=True)
        df[f"{col}_rolling_std"] = df.groupby("unit_number")[col].rolling(window=window_size,
                                                                          min_periods=1).std().reset_index(0, drop=True)

    return df


def add_lag_features(df, lag=1):
    """Add lag features (previous cycle sensor values)"""
    sensor_cols = [col for col in df.columns if col.startswith("sensor_")]

    for col in sensor_cols:
        df[f"{col}_lag{lag}"] = df.groupby("unit_number")[col].shift(lag)

    # Fill NaN values (first cycle doesn't have previous data)
    df.fillna(method="bfill", inplace=True)

    return df


def process_feature_engineering(input_file, output_file):
    """Main function to apply feature engineering"""
    print("Loading dataset for feature engineering...")
    df = load_data(input_file)

    print("Applying rolling statistics...")
    df = add_rolling_features(df)

    print("Applying lag features...")
    df = add_lag_features(df)

    # Extract feature columns (excluding 'unit_number', 'time_in_cycles', 'RUL')
    feature_cols = [col for col in df.columns if col not in ["unit_number", "time_in_cycles", "RUL"]]

    print("Refitting MinMaxScaler on the new feature set...")
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])  # Fit and transform

    # Save processed dataset
    df.to_csv(output_file, index=False)
    print(f"Feature-engineered dataset saved to {output_file}")


if __name__ == "__main__":
    train_file = "D:/ProjectML/Engine_Failure/my_project/data/processed/train_cleaned.csv"
    train_output = "D:/ProjectML/Engine_Failure/my_project/data/processed/train_features.csv"

    print("Processing training set...")
    process_feature_engineering(train_file, train_output)
