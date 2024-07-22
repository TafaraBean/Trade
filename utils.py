import pandas as pd
import os
import MetaTrader5 as mt5
from typing import IO,List


def add_missing_position(order: pd.Series, file_path: str, columns: List[str] = ['ticket', 'sl', 'tp', 'be','be_condition']) -> None:
    # Convert Series to DataFrame
    if order['type'] == mt5.ORDER_TYPE_BUY:
        if order['sl'] > order['price_open']:
            order['be_condition'] = order['sl'] + 10 * 0.0001
            order['be'] = order['sl'] + 8 * 0.0001

        else:
            order['be_condition'] = order['price_open'] + 10 * 0.0001

    else:
        if order['sl'] < order['price_open']:
            order['be_condition'] = order['sl'] - 10 * 0.0001
            order['be'] = order['sl'] - 8 * 0.0001

        else:
            order['be_condition'] = order['price_open'] - 10 * 0.0001
            order['be'] = order['price_open']- 8 *0.0001

    order_df = order.to_frame().T
    
    # Select only the specified columns that are present in the DataFrame
    order_subset = order_df[[col for col in columns if col in order_df.columns]]
    
    # Write the DataFrame with selected columns to the CSV file
    order_subset.to_csv(file_path, 
                        mode='a',
                        header=False, 
                        index=False)
    

