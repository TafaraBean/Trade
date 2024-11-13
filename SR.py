import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from main import bot

start = pd.Timestamp("2024-06-01")
end = pd.Timestamp("2024-06-30")

# Pulling the data from MT5
data = bot.copy_chart_range(symbol=bot.symbol, timeframe=bot.timeframe, start=start, end=end)
# Sample data (replace with your closing prices data)
closing_prices = data['close']

# K-Means for clustering price levels
prices = np.array(np.unique(closing_prices)).reshape(-1, 1)
kmeans = KMeans(n_clusters=10)  # Choose cluster count based on data
kmeans.fit(prices)
sr_levels = sorted(kmeans.cluster_centers_.flatten())

# Plot Closing Prices and Support/Resistance Zones
plt.figure(figsize=(10, 6))
plt.plot(closing_prices, label="Closing Prices", color="black")
for level in sr_levels:
    plt.axhline(y=level, color="blue", linestyle="--", alpha=0.7, label="Clustered S/R Level" if level == sr_levels[0] else "")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.title("Closing Prices with Clustered Support/Resistance Levels")
plt.show()
