from analysis import * 

# Load data
data = pd.read_csv('beta/BTCUSDT86400.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')


# Load data
#data = pd.read_csv('beta/output.csv')
#data['time'] = data['time'].astype('datetime64[s]')
#data = data.set_index('time')


# Take natural log of data to resolve price scaling issues
data = np.log(data)
# Trendline parameter
lookback = 30


support_slope = [np.nan] * len(data)
resist_slope = [np.nan] * len(data)
for i in range(lookback - 1, len(data)):
    candles = data.iloc[i - lookback + 1: i + 1]
    support_coefs, resist_coefs =  fit_trendlines_high_low(candles['high'], 
                                                           candles['low'], 
                                                           candles['close'])
    support_slope[i] = support_coefs[0]
    resist_slope[i] = resist_coefs[0]

data['support_slope'] = support_slope
data['resist_slope'] = resist_slope

print(data)
import plotly.graph_objects as go


min_value = data['close'].min()

# Create a candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close']
)])

# Calculate start and end points for the support and resistance lines
start_date = data.index[0]
end_date = data.index[-1]
start_close = data['close'].iloc[0]
end_close = data['close'].iloc[-1]

# Support line
support_slope = data['support_slope'].dropna().iloc[0]
support_start = start_close - 3
support_end = data['close'].iloc[-1] - 3#support_start + support_slope * (len(data) - 1)

# Resistance line
resist_slope = data['resist_slope'].dropna().iloc[0]
resist_start = start_close
resist_end = resist_start + resist_slope * (len(data) - 1)

# Add support line
fig.add_trace(go.Scatter(
    x=[start_date, end_date],
    y=[min_value, support_end],
    mode='lines',
    name='Support Line',
    line=dict(color='green')
))

# Add resistance line
fig.add_trace(go.Scatter(
    x=[start_date, end_date],
    y=[resist_start, resist_end],
    mode='lines',
    name='Resistance Line',
    line=dict(color='red')
))

# Update layout
fig.update_layout(
    title='Candlestick Chart with Support and Resistance Lines',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,
    template='plotly_dark'
)

# Set the layout for the plot
fig.update_layout(
    title='Candlestick Chart',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=False,  # Hide the range slider
    template='plotly_dark'  # Dark theme
)

# Show the figure
fig.show()