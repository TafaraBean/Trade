from datetime import datetime, timedelta


def time_until_next_boundary(time_frame):
  """
  Calculates the seconds until the next time boundary (hour, half-hour, quarter-hour, 5 minutes, or 1 minute).

  Args:
      time_frame (str): "hour", "half_hour", "quarter_hour", "five_minutes", or "one_minute".

  Returns:
      tuple: (int, datetime): Seconds until next boundary and the datetime object with added seconds.
  """

  now = datetime.now()
  hour = now.hour
  minute = now.minute

  if time_frame == "hour":
      # Calculate next hour in minutes (considering potential overflow)
      next_minute = (hour + 1) * 60

      # Subtract current minute and convert to seconds
      seconds_until_boundary = (next_minute - minute) * 60

  elif time_frame == "half_hour":
      # Calculate the target minute (0 or 30) for the next half hour
      target_minute = (minute // 30 + 1) * 30

      # Calculate seconds until target minute
      seconds_until_boundary = (target_minute - minute) * 60

  elif time_frame == "quarter_hour":
      # Similar logic to half-hour, but target minute is 0, 15, 30, or 45
      target_minute = (minute // 15 + 1) * 15
      seconds_until_boundary = (target_minute - minute) * 60

  elif time_frame == "five_minutes":
      target_minute = (minute // 5 + 1) * 5
      seconds_until_boundary = (target_minute - minute) * 60

  elif time_frame == "one_minute":
      seconds_until_boundary = (60 - minute) * 60

  else:
      raise ValueError("Invalid time frame. Choose 'hour', 'half_hour', 'quarter_hour', 'five_minutes', or 'one_minute'.")

  # Add seconds to current time
  time_with_added_seconds = now + timedelta(seconds=seconds_until_boundary)

  return seconds_until_boundary, time_with_added_seconds


# Example usage
time_frames = ["hour", "half_hour", "quarter_hour", "five_minutes", "one_minute"]
for time_frame in time_frames:
  seconds_until_next, future_time = time_until_next_boundary(time_frame)
  print(f"Seconds until next {time_frame} boundary: {seconds_until_next}")
  print(f"Time with added seconds: {future_time.strftime('%H:%M:%S')}")
  print()  # Add an empty line for better readability
