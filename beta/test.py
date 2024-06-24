import pandas as pd
import numpy as np
import plotly.graph_objects as go
from analysis import fit_trendlines_high_low  # Ensure this function is correctly imported from your analysis module

# Load data
data = pd.read_csv('beta/output.csv')
data['time'] = data['time'].astype('datetime64[s]')
data = data.set_index('time', drop=False)

# Take natural log of data to resolve price scaling issues
df_log = np.log(data[['high', 'low', 'close']])

# Trendline parameter
lookback = 15

# Initialize columns for trendlines and their gradients
data['support_trendline'] = np.nan
data['resistance_trendline'] = np.nan
data['support_gradient'] = np.nan
data['resistance_gradient'] = np.nan

# Iterate over the dataset in overlapping windows of 15 candles
for i in range(lookback, len(df_log) + 1):
    window_data = df_log.iloc[i - lookback:i]
    support_coefs, resist_coefs = fit_trendlines_high_low(window_data['high'], window_data['low'], window_data['close'])

    # Extract slope and intercept
    support_slope, support_intercept = support_coefs
    resist_slope, resist_intercept = resist_coefs

    # Apply the calculated gradients to each candle in the window
    for j in range(lookback):
        idx = i - lookback + j
        support_value = support_slope * j + support_intercept
        resist_value = resist_slope * j + resist_intercept
        data.at[data.index[idx], 'support_trendline'] = np.exp(support_value)
        data.at[data.index[idx], 'resistance_trendline'] = np.exp(resist_value)
        data.at[data.index[idx], 'support_gradient'] = support_slope
        data.at[data.index[idx], 'resistance_gradient'] = resist_slope

data.to_csv("beta/test.csv",index=False)
# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close']
)])

# Add support and resistance lines
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['support_trendline'],
    mode='lines',
    name='Support Line',
    line=dict(color='green'),
    connectgaps=False
))

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['resistance_trendline'],
    mode='lines',
    name='Resistance Line',
    line=dict(color='red'),
    connectgaps=False
))

# Update layout
fig.update_layout(
    title='Candlestick Chart with Support and Resistance Lines',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Show the figure
fig.show()
    
  