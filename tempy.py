def calculate_profit_loss(lot_size, open_price, close_price, is_buy):
  """
  This function calculates the profit or loss for a given trade.

  Args:
      lot_size: The number of units bought or sold (e.g., shares, contracts).
      open_price: The price at which the trade was opened.
      close_price: The price at which the trade was closed.
      is_buy: A boolean flag indicating whether the trade was a buy (True) or sell (False).

  Returns:
      The profit or loss for the trade. A positive value indicates profit, 
      and a negative value indicates loss.
  """

  if is_buy:
    # Profit for buy trade = (close price - open price) * lot size
    profit_loss = (close_price - open_price) * lot_size
  else:
    # Profit for sell trade = (open price - close price) * lot size
    profit_loss = (open_price - close_price) * lot_size

  return profit_loss

# Example usage
lot_size = 0.02
open_price = 62040.84
close_price = 59612.85815427919
is_buy = False  # Buy trade

profit_loss = calculate_profit_loss(lot_size, open_price, close_price, is_buy)

if profit_loss > 0:
  print("Profit:", profit_loss)
elif profit_loss < 0:
  print("Loss:", abs(profit_loss))  # Print absolute value for loss
else:
  print("No profit or loss (break-even)")
