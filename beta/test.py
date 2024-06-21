def calculate_progress_towards_goal(entry_price, current_price, goal_price):
  """
  Calculates the percentage progress towards the goal price from the entry price.

  Args:
      entry_price (float): The price at which you entered the trade.
      current_price (float): The current price of the asset.
      goal_price (float): The target price you're aiming for (profit or loss).

  Returns:
      float: The percentage progress towards the goal price (0.0 to 1.0 or -1.0 to 0.0),
          positive for profit, negative for loss.
  """
  if entry_price == 0:
    raise ZeroDivisionError("Entry price cannot be zero.")
  price_change = current_price - entry_price
  progress = price_change / (goal_price - entry_price)

  # Handle cases where goal price is equal to entry price
  if abs(goal_price - entry_price) < 1e-6:
    progress = 0.0

  return progress * 100

# Example usage
entry_price = 200
current_price = 120
goal_price = 100  # Aiming for profit

progress = calculate_progress_towards_goal(entry_price, current_price, goal_price)
print(f"Progress towards goal price: {progress:.2f}%")

# Example with loss scenario
goal_price = 80  # Aiming for loss (selling at a lower price)
progress = calculate_progress_towards_goal(entry_price, current_price, goal_price)
print(f"Progress towards goal price: {progress:.2f}%")
