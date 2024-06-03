import MetaTrader5 as mt5



# Main settings

account_id = 1234567890
password=''
server=''



# Symbol settings
symbol = 'XAUUSD'
lot = 0.1
add_lot = 0.01
min_deleverage = 15
deleverage_steps = 7
take_profit_short = 21
action = mt5.TRADE_ACTION_DEAL
order_type = mt5.ORDER_TYPE_BUY

stop_loss = 1
take_profit = 10




# Init
if not mt5.initialize():
    print('initialize() failed, error code =', mt5.last_error())
    quit()

mt5.login(account_id, password, server)
# Timeframe settings
timeframe = mt5.TIMEFRAME_M1

request = {
    'action': action,
    'symbol': symbol,
    'volume': 0.1,
    'type': order_type,
    'price': 0,
    'sl': stop_loss,
    'tp': take_profit,
    'deviation': 20,
    'magic':0,
    'comment': 'python market order',
    'type_time': mt5.ORDER_TIME_GTC,
    'type_filling': mt5.ORDER_FILLING_IOC,
}

response =  mt5.order_send(request)
print(response)