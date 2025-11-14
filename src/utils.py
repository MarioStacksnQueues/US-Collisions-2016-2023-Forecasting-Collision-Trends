"""
Utility functions for US Collisions Forecasting Project
Author: Mario Cuevas
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def load_daily_features(file_path: str = '../data/processed/daily_accidents_features.csv') -> pd.DataFrame:
    """
    Load the daily accident features dataset.

    Args:
        file_path: Path to the daily_accidents_features.csv file

    Returns:
        DataFrame with parsed dates
    """
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def create_rolling_features(df: pd.DataFrame,
                           column: str,
                           windows: List[int] = [7, 30]) -> pd.DataFrame:
    """
    Create rolling mean and std features for a given column.

    Args:
        df: Input DataFrame
        column: Column name to create rolling features for
        windows: List of window sizes for rolling calculations

    Returns:
        DataFrame with added rolling features
    """
    df_copy = df.copy()

    for window in windows:
        df_copy[f'{column}_{window}d_MA'] = df_copy[column].rolling(
            window=window, min_periods=1
        ).mean()

        df_copy[f'{column}_{window}d_Std'] = df_copy[column].rolling(
            window=window, min_periods=1
        ).std()

    return df_copy


def create_lag_features(df: pd.DataFrame,
                       column: str,
                       lags: List[int] = [1, 3, 7, 14, 30]) -> pd.DataFrame:
    """
    Create lagged features for a given column.

    Args:
        df: Input DataFrame
        column: Column name to create lag features for
        lags: List of lag periods

    Returns:
        DataFrame with added lag features
    """
    df_copy = df.copy()

    for lag in lags:
        df_copy[f'{column}_Lag_{lag}'] = df_copy[column].shift(lag)

    return df_copy


def get_season(month: int) -> int:
    """
    Convert month number to season.

    Args:
        month: Month number (1-12)

    Returns:
        Season number (1=Winter, 2=Spring, 3=Summer, 4=Fall)
    """
    if month in [12, 1, 2]:
        return 1  # Winter
    elif month in [3, 4, 5]:
        return 2  # Spring
    elif month in [6, 7, 8]:
        return 3  # Summer
    else:
        return 4  # Fall


def create_weather_risk_index(df: pd.DataFrame,
                              temp_col: str = 'Avg_Temperature',
                              humidity_col: str = 'Avg_Humidity',
                              visibility_col: str = 'Avg_Visibility',
                              wind_col: str = 'Avg_Wind_Speed',
                              precip_col: str = 'Avg_Precipitation') -> pd.DataFrame:
    """
    Create a composite weather risk index.

    Higher values indicate more dangerous weather conditions.

    Args:
        df: Input DataFrame
        temp_col: Temperature column name
        humidity_col: Humidity column name
        visibility_col: Visibility column name
        wind_col: Wind speed column name
        precip_col: Precipitation column name

    Returns:
        DataFrame with added Weather_Risk_Index column
    """
    df_copy = df.copy()

    # Standardize weather features
    weather_features = [temp_col, humidity_col, visibility_col, wind_col, precip_col]
    scaler = StandardScaler()

    weather_scaled = scaler.fit_transform(df_copy[weather_features])

    # Create composite index (higher = worse conditions)
    weather_risk_index = (
        0.3 * (-weather_scaled[:, 2]) +  # Low visibility is dangerous
        0.3 * weather_scaled[:, 4] +     # High precipitation is dangerous
        0.2 * weather_scaled[:, 3] +     # High wind is dangerous
        0.2 * weather_scaled[:, 1]       # High humidity correlates with accidents
    )

    df_copy['Weather_Risk_Index'] = weather_risk_index

    return df_copy


def split_time_series(df: pd.DataFrame,
                      train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.

    Args:
        df: Input DataFrame (should be sorted by date)
        train_ratio: Ratio of data to use for training

    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray) -> dict:
    """
    Calculate regression metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, MAPE, and R² metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def prepare_prophet_data(df: pd.DataFrame,
                        date_col: str = 'Date',
                        target_col: str = 'Accident_Count',
                        additional_regressors: List[str] = None) -> pd.DataFrame:
    """
    Prepare data for Facebook Prophet model.

    Args:
        df: Input DataFrame
        date_col: Name of date column
        target_col: Name of target column
        additional_regressors: List of additional columns to include as regressors

    Returns:
        DataFrame formatted for Prophet (with 'ds' and 'y' columns)
    """
    prophet_df = df[[date_col, target_col]].copy()
    prophet_df.columns = ['ds', 'y']

    if additional_regressors:
        for reg in additional_regressors:
            prophet_df[reg] = df[reg].values

    return prophet_df


def extract_temporal_features(df: pd.DataFrame,
                              datetime_col: str = 'Start_Time') -> pd.DataFrame:
    """
    Extract temporal features from a datetime column.

    Args:
        df: Input DataFrame
        datetime_col: Name of datetime column

    Returns:
        DataFrame with added temporal features
    """
    df_copy = df.copy()

    # Ensure datetime
    df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col])

    # Extract features
    df_copy['Year'] = df_copy[datetime_col].dt.year
    df_copy['Month'] = df_copy[datetime_col].dt.month
    df_copy['Day'] = df_copy[datetime_col].dt.day
    df_copy['DayOfWeek'] = df_copy[datetime_col].dt.dayofweek  # 0=Monday
    df_copy['Hour'] = df_copy[datetime_col].dt.hour
    df_copy['Date'] = df_copy[datetime_col].dt.date

    # Derived features
    df_copy['Is_Weekend'] = (df_copy['DayOfWeek'] >= 5).astype(int)
    df_copy['Is_Rush_Hour'] = (
        ((df_copy['Hour'] >= 7) & (df_copy['Hour'] <= 9)) |
        ((df_copy['Hour'] >= 16) & (df_copy['Hour'] <= 19))
    ).astype(int)

    df_copy['Season'] = df_copy['Month'].apply(get_season)

    return df_copy


if __name__ == '__main__':
    # Example usage
    print("US Collisions Forecasting - Utility Functions")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - load_daily_features()")
    print("  - create_rolling_features()")
    print("  - create_lag_features()")
    print("  - create_weather_risk_index()")
    print("  - split_time_series()")
    print("  - calculate_metrics()")
    print("  - prepare_prophet_data()")
    print("  - extract_temporal_features()")
    print("  - get_season()")
    print("\nFor documentation, use: help(function_name)")
