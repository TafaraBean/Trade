import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_linear_regression_channels(data, lookback, high_col, low_col, close_col):
    """
    Calculate linear regression channels for a rolling window.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing price data.
    - lookback (int): The rolling window size.
    - high_col (str): The column name for high prices.
    - low_col (str): The column name for low prices.
    - close_col (str): The column name for close prices.

    Returns:
    - data (pd.DataFrame): The DataFrame with added columns for support and resistance gradients and values.
    """
    # Ensure the input DataFrame has the necessary columns
    required_cols = [high_col, low_col, close_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Add columns to store the results
    data[f'fixed_resistance_gradient_{lookback}'] = np.nan
    data[f'fixed_support_gradient_{lookback}'] = np.nan
    data[f'fixed_support_trendline_{lookback}'] = np.nan
    data[f'fixed_resistance_trendline_{lookback}'] = np.nan

    for i in range(lookback, len(data)):
        # Extract rolling window data
        window_data = data.iloc[i - lookback:i]
        X = np.arange(len(window_data)).reshape(-1, 1)  # Time indices for regression

        # Fit regression lines for high and low prices
        high_model = LinearRegression().fit(X, window_data[high_col])
        low_model = LinearRegression().fit(X, window_data[low_col])

        # Extract slopes (gradients) and intercepts
        resist_slope, resist_intercept = high_model.coef_[0], high_model.intercept_
        support_slope, support_intercept = low_model.coef_[0], low_model.intercept_

        # Calculate current support and resistance values based on the close price
        current_close = data[close_col].iloc[i]
        support_value = support_slope * (lookback - 1) + support_intercept
        resist_value = resist_slope * (lookback - 1) + resist_intercept

        # Update the results in the DataFrame
        data.at[data.index[i], f'fixed_resistance_gradient_{lookback}'] = resist_slope
        data.at[data.index[i], f'fixed_support_gradient_{lookback}'] = support_slope
        data.at[data.index[i], f'fixed_support_trendline_{lookback}'] = support_value
        data.at[data.index[i], f'fixed_resistance_trendline_{lookback}'] = resist_value

    return data

# Example usage:
# df = pd.DataFrame({"high": [...], "low": [...], "close": [...]})
# lookback_period = 15
# df_with_channels = calculate_linear_regression_channels(df, lookback_period, "high", "low", "close")