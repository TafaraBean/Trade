def analyze(df):
    #add stop loss and take proft
    df["sl"] = None
    df["tp"] = None
    for index in range(len(df)):
        if df.at[index, 'is_buy2']:
            df.at[index, "tp"] = df.at[index, 'low'] + 9 
            df.at[index, "sl"] = df.at[index, 'low'] - 3 
        elif df.at[index, 'is_sell2']:
            df.at[index, "tp"] = df.at[index, 'high'] - 9 
            df.at[index, "sl"] = df.at[index, 'high'] + 3 

    #calculate with tp or sl hit first
