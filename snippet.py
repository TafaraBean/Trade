import pandas as pd

# Get current time
current_time = pd.Timestamp.now()
next_interval = current_time.ceil("10min")

# Calculate the difference in seconds
time_difference = (next_interval - current_time).total_seconds()

print(f"Current time: {current_time}")
print(f"Next 30-minute interval: {next_interval}")
print(f"Seconds until next 30-minute interval: {time_difference}")
