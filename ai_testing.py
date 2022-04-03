from os import close
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
symbol='BTCUSDT'
prediction_candles=60   
candles = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
#ohlcv are values returned from get histrical klines
all_closes=[]
for kline in candles:
    all_closes.append((float(kline[4])))
all_closes=np.array(all_closes)
closes=all_closes
closes=np.array(closes)
closes=closes[int(len(closes)*1/3):int(len(closes)*2/3)]
test_data_2=all_closes[int((len(all_closes)*2)/3):]
actual_price=all_closes[int((len(all_closes)*2)/3):]
max_val=0
best_prediction_candles=10
candles_into_future=9
fee= testing_functions.calc_fee(symbol)
for x in range(10,90):
    prediction_candles = x   
    x2 = ai_testing_functions.get_x_train(closes,prediction_candles,candles_into_future)    
    y2 = ai_testing_functions.get_y_train(closes,prediction_candles,candles_into_future)
    model_2 = ai_testing_functions.build_model(x2,y2)
    prediction_prices_2 = ai_testing_functions.get_model_predictions(model_2,test_data_2,prediction_candles)
    prices = all_closes[int((len(all_closes)*2)/3)+prediction_candles-1:]
    # model.save('my_model.h5')
    # del model
    # model = load_model('my_model.h5')
    actual_price=actual_price[prediction_candles+1:]#cuts the model input
    #prediction_prices=prediction_prices[8:]
    prediction_prices_2=prediction_prices_2[candles_into_future-1:]
    curr_val=float(testing_functions.check_profits_test_const_fee(prediction_prices_2,prices,symbol,1,1,fee,False))
    if(curr_val>max_val):
        max_val= curr_val
        best_prediction_candles=prediction_candles
        print(f'the best value so far is {max_val} using {best_prediction_candles} prediction candles')

print(f'the best value so far is {max_val} using {best_prediction_candles} prediction candles')

# plt.plot(actual_price , color='red', label='actual price')
# plt.plot(prediction_prices , color='blue',label='ai')
# plt.plot(prediction_prices_2 , color='yellow',label='prediction_2')
# plt.title(f'{symbol} price prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend(loc='upper left')
# plt.show()

