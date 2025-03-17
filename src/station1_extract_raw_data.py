import requests
from datetime import datetime
import pandas as pd
import numpy as np
import logging

def station1_load_data(crypto_pairs, api_key,limit, filepath):
    def fetch_crypto_data(symbol, api_key, limit):
        if api_key.strip():  # Check if api_key is not empty or just whitespace
            headers = {'Apikey': api_key}  # The header with your API key
        else:
            headers = {}  # Empty headers if api_key is empty

        url = f'https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={limit}'    
        response = requests.get(url, headers=headers)
        data = response.json()
        return data['Data']['Data']
   
    # Create an empty DataFrame to store the data
    df = pd.DataFrame()
    # Fetch historical data for each pair
    for symbol in crypto_pairs:
        print(f"Fetching data for {symbol}...")
        data = pd.DataFrame( fetch_crypto_data(symbol,api_key=api_key,limit=limit))     
        data ['date'] = pd.to_datetime(data ['time'], unit='s')
        data.drop(['time'], inplace = True, axis = 1)
        data['ticker'] = symbol
        df = pd.concat([df,data], axis = 0)   
   
    if not df.empty: 
        # Save the data to a CSV file at the specified filepath
        dfStation1 = df.set_index(['ticker','date'])
        dfStation1 = dfStation1.sort_index(level = ['ticker','date'])
        dfStation1 = dfStation1.replace([np.inf, -np.inf], np.nan)
        dfStation1.to_csv(filepath)
        logging.info("____")
        logging.info(f"Data successfully saved to {filepath}")
    else:
        logging.warning("No data fetched. CSV file not created.")
    return df

def top_market_cap_list():
    top_crypto = 10
    url = f'https://min-api.cryptocompare.com/data/top/mktcapfull?limit={top_crypto}&tsym=USD'
    response = requests.get(url)
    data = response.json()    
    pairs = data['Data']
    df = pd.DataFrame(pairs)
    df['Name'] = df['CoinInfo'].apply(lambda x: x['Name'])
    return df['Name'].values.tolist()

top_crypto_list = top_market_cap_list()

api_key= ''
limit  = 365
filepath= r'/Users/Johnny/Desktop/crypto-portfolio-data-engineering-project/data/stage_1_crypto_data.csv' # Change this to your filepath

##################### Execute the function ###################
df = station1_load_data(top_crypto_list,api_key,limit,filepath)
##############################################################
########################### END ##############################