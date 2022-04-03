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
prediction_candles_1=60
prediction_candles_2=60   
candles = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")
#ohlcv are values returned from get histrical klines
all_closes=[]
for kline in candles:
    all_closes.append((float(kline[4])))
all_closes=np.array(all_closes)

closes=all_closes
closes=np.array(closes)
closes=closes[int(len(closes)*1/3):int(len(closes)*2/3)]
test_data=all_closes[int((len(all_closes)*2)/3):-1]
test_data_2=all_closes[int((len(all_closes)*2)/3):-2]
actual_price=test_data
actual_price_2=test_data_2
candles_into_future=9

x = ai_testing_functions.get_x_train(closes,prediction_candles_1,1)
x2 = ai_testing_functions.get_x_train(closes,prediction_candles_2,candles_into_future)
y = ai_testing_functions.get_y_train(closes,prediction_candles_1,1)
y2 = ai_testing_functions.get_y_train(closes,prediction_candles_2,candles_into_future)

model_2 = ai_testing_functions.build_model(x2,y2)
prediction_prices_2 = ai_testing_functions.get_model_predictions(model_2,test_data,prediction_candles_2)
prices_2 = all_closes[int((len(all_closes)*2)/3)+prediction_candles_2-1:]
fee=testing_functions.calc_fee(symbol)
# model.save('my_model.h5')
# del model
# model = load_model('my_model.h5')

actual_price_2=actual_price_2[prediction_candles_2+1:]

prediction_prices_2=prediction_prices_2[candles_into_future-1:]#for some reason the n-th day is what will      
print(testing_functions.check_profits_test_const_fee(prediction_prices_2,prices_2,symbol,1,1,fee,False))
#plt.plot(actual_price , color='red', label='actual price 1')     #actually happend on the what is supposed to be the first prediction
plt.plot(actual_price_2 , color='black', label='actual price 2')
plt.plot(prediction_prices_2 , color='green',label='ai_2')
plt.title(f'{symbol} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

# print(testing_functions.total_gain_possible(actual_price))
# print(testing_functions.total_loss_possible(actual_price))