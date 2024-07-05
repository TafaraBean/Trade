def calc_profit_factor(loss: float,
                       gross_profit: float):
    if loss != 0:
        profit_factor = gross_profit / abs(loss)
    else:
        profit_factor = float('inf')  # Handle case where there are no losing trades
    
    return profit_factor

def calc_percentage_profitability(successful_trades: int,
                       break_even: int,
                       total_trades: int):
    if total_trades > 0:
        percentage_profitability = ((successful_trades+break_even) / (total_trades)) * 100
    else:
        percentage_profitability = 0  # Handle case where there are no trades
    
    return percentage_profitability

profit_factor = calc_profit_factor(gross_profit=gross_profit,
                                    loss=loss)
percentage_profitability = calc_percentage_profitability(successful_trades=successful_trades,
                                                            break_even=break_even,
                                                            total_trades=total_trades)
