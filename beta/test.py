def calculate_percentage_completion(entry_price, goal_price, current_price, is_buy):
  """
  Calculates the percentage completion of a running trade.

  Args:
      entry_price (float): The price at which the trade was entered.
      goal_price (float): The target price for the trade.
      current_price (float): The current market price of the asset.
      is_buy (bool): True if it's a buy trade, False if it's a sell trade.

  Returns:
      float: The percentage completion of the trade (0.0 to 1.0).
  """

  if entry_price == goal_price:
    return 1.0  # Handle case where entry and goal prices are equal

  # Calculate the price movement direction
  price_movement = current_price - entry_price

  # Adjust calculation based on buy or sell trade
  if is_buy:
    completion_ratio = price_movement / (goal_price - entry_price)
  else:
    completion_ratio = (entry_price - current_price) / (entry_price - goal_price)

  # Clamp the completion ratio to the range [0.0, 1.0]
  completion_ratio = max(0.0, min(completion_ratio, 1.0))

  return completion_ratio * 100  # Convert to percentage and return


# Example usage
entry_price = 100.0
goal_price = 200
current_price = 99
is_buy = True  # Buy trade

percentage_completion = calculate_percentage_completion(entry_price, goal_price, current_price, is_buy)
print(f"Percentage completion for buy trade: {percentage_completion:.2f}%")

# Example for sell trade
entry_price = 200
goal_price = 100.0
current_price = 17
is_buy = False  # Sell trade

percentage_completion = calculate_percentage_completion(entry_price, goal_price, current_price, is_buy)
print(f"Percentage completion for sell trade: {percentage_completion:.2f}%")
