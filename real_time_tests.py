from os import close
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import binance
import time
from tensorflow.python.framework.auto_control_deps import _ORDER_INSENSITIVE_STATEFUL_OPS
from tensorflow.python.keras.backend import shape
from tensorflow.python.keras.engine import sequential
import config,testing_functions,ai_testing_functions,real_time_test_functions,client as bin_client

from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python import client
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from tensorflow.python.keras.saving.save import load_model

from binance.client import Client
client = bin_client.Client(config.API_KEY, config.SECRET_KEY)
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
test_data=all_closes[int((len(all_closes)*2)/3):]
test_data_2=all_closes[int((len(all_closes)*2)/3):]
test_data_2 = real_time_test_functions.get_training_data(all_closes,prediction_candles)
actual_price=all_closes[-prediction_candles*2]
candles_into_future=9

x2 = ai_testing_functions.get_x_train(closes,prediction_candles,candles_into_future)
y2 = ai_testing_functions.get_y_train(closes,prediction_candles,candles_into_future)
model_2 = ai_testing_functions.build_model(x2,y2)

prediction_prices_2 = ai_testing_functions.get_model_predictions(model_2,test_data,prediction_candles)
# to del -> prices = all_closes[int((len(all_closes)*2)/3)+prediction_candles-1:]
## up until here we build the model as usual
## from here we need to rebuild it every new candle ? :/
closes1= all_closes
closes2=[]
temp= client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 hour ago UTC")#initialise
for kline in temp:
    closes2.append((float(kline[4])))

new_candles=[]
predictions=[]
new_candles_count=0
to_log=True

while (new_candles_count<3):
    while (not real_time_test_functions.is_new_candle(closes1,closes2)):
        if(to_log):print("waiting for new candles")
        time.sleep(1)
        closes2=[]
        temp= client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 hour ago UTC")#initialise
        for kline in temp:
            closes2.append((float(kline[4])))        
    new_candles.append(closes2[-1])
    if(to_log):print("got new candle")
    predictions.append(ai_testing_functions.get_model_predictions(model_2,closes2[-prediction_candles:],prediction_candles))
    real_time_test_functions.log(closes2[-2],closes[-1],predictions[-1])
    new_candles_count+=1

plt.plot(new_candles , color='red', label='actual candles')
#plt.plot(prediction_prices , color='blue',label='ai')
plt.plot(predictions , color='yellow',label='ai')
plt.title(f'{symbol} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()