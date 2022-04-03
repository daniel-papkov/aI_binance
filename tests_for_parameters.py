from os import close
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import binance
from tensorflow.python.framework.auto_control_deps import _ORDER_INSENSITIVE_STATEFUL_OPS
from tensorflow.python.keras.backend import shape
from tensorflow.python.keras.engine import sequential
import config,testing_functions,ai_testing_functions

from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python import client
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from tensorflow.python.keras.saving.save import load_model

from binance.client import Client
client = Client(config.API_KEY, config.SECRET_KEY)
symbol='CAKEBNB'
prediction_candles=60   
candles = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
#ohlcv are values returned from get histrical klines
all_closes=[]
for kline in candles:
    all_closes.append((float(kline[4])))
closes=all_closes
closes=np.array(closes)
closes=closes[int(len(closes)*1/3):int(len(closes)*2/3)]
test_data=all_closes[int((len(all_closes)*2)/3):-1]
test_data_2=all_closes[int((len(all_closes)*2)/3):-2]
test_data_2= np.array(test_data_2)
actual_price=test_data
candles_into_future=6

x2 = ai_testing_functions.get_x_train(closes,prediction_candles,candles_into_future)
y2 = ai_testing_functions.get_y_train(closes,prediction_candles,candles_into_future)
model_2 = ai_testing_functions.build_model(x2,y2)
prediction_prices_2 = ai_testing_functions.get_model_predictions(model_2,test_data_2,prediction_candles)
prices = all_closes[int((len(all_closes)*2)/3)+prediction_candles-1:]
actual_price=actual_price[prediction_candles+1:]
prediction_prices_2=prediction_prices_2[candles_into_future-1:]#for some reason the n-th day is what will      #actually happend on the what is supposed to be the first prediction
results=[]
fee= testing_functions.calc_fee(symbol)
max_res=0
max_buy=0
max_sell=0
for i in range(0,50):
    max_res=0
    max_i_index=0
    max_j_index=0
    parameters=[]
    holdbuy_percent,sellignore_percent=0,0
    holdbuy_percent=1+(i/10000)
    for j in range(0,50):
        res=0.0
        sellignore_percent=((holdbuy_percent -1.0) / 3.0) + 1.0
        #print(sellignore_percent)
        res=float(testing_functions.check_profits_test_const_fee(prediction_prices_2,prices,symbol,holdbuy_percent,sellignore_percent,fee, False))
        #print(res)
        results.append(res)

max_res=max(results)
i=results.index(max_res) / 500
j=results.index(max_res) % 500
max_buy=1+(i/10000)
max_sell=1-(j/10000)
print(f'the max value is {max_res}')
print(f'the beset paremeters:')
print(f'buy:{max_buy}')
print(f'sell:{max_sell}')
trade_fee=testing_functions.calc_fee(symbol)
zibi_tocaa = testing_functions.check_profits_test(prediction_prices_2,prices,symbol,1.0,1.0, trade_fee)
zibi_only=testing_functions.check_profits_test(prediction_prices_2,prices,symbol,1.0,1.0,1)
print(f'IMashma bli amla {zibi_only}')
print(f' im amla {zibi_tocaa}')
