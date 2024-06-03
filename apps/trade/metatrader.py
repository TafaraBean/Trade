import MetaTrader5 as mt5



# Main settings

account_id = 1234567890
password=''
server=''



# Symbol settings
symbol = 'EURUSD'
sl_multiplier = 13

lot = 0.1
add_lot = 0.01
min_deleverage = 15
deleverage_steps = 7
take_profit_short = 21
sl_short = take_profit_short * sl_multiplier


# Init
if not mt5.initialize():
    print('initialize() failed, error code =', mt5.last_error())
    quit()

mt5.login(account_id, password, server)
# Timeframe settings
timeframe = mt5.TIMEFRAME_M1

selected = mt5.symbol_select(symbol)
if not selected:
    print('symbol_select({}) failed, error code = {}'.format(symbol, mt5.last_error()))
    quit()


def get_position_data():
    positions=mt5.positions_get(symbol=symbol)
    # print(positions)
    if positions == None:
        print(f'No positions on {symbol}')
    elif len(positions) > 0:
        # print(f'Total positions on {symbol} =',len(positions))
        for position in positions:
            post_dict = position._asdict()
            global pos_price, identifier, volume
            pos_price = post_dict['price_open']
            identifier = post_dict['identifier']
            volume = post_dict['volume']
            print(pos_price, identifier, volume)
