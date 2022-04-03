import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import binance
from tensorflow.python.framework.auto_control_deps import _ORDER_INSENSITIVE_STATEFUL_OPS
from tensorflow.python.keras.backend import shape
import config,testing_functions


from sklearn.preprocessing import  MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.python import client
from tensorflow.python.keras.saving.hdf5_format import load_model_from_hdf5
from tensorflow.python.keras.saving.save import load_model

from binance.client import Client

def build_model(x_train,y_train):        
    model= Sequential()
    model.add(LSTM(units=100,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100,return_sequences=True)) 
    model.add(Dropout(0.2))
    model.add(LSTM(units=100,return_sequences=True)) 
    model.add(Dropout(0.2))
    model.add(LSTM(units=100,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(x_train,y_train,epochs=25,batch_size=32)
    return model

def get_x_train(closes,prediction_candles,days_into_future):
    x_train=[]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(closes.reshape(-1,1))
    for x in range(0,len(scaled_data)-prediction_candles-days_into_future):
        x_train.append(scaled_data[x:x+prediction_candles,0])#get the history
    x_train= np.array(x_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    return x_train

def get_y_train(closes,prediction_candles,days_into_future):#days into the future to learn based on X_train
    y_train=[]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(closes.reshape(-1,1))
    for x in range(0,len(scaled_data)-prediction_candles-days_into_future):
        y_train.append(scaled_data[x+prediction_candles + days_into_future,0])#get the history
    y_train= np.array(y_train)    
    return y_train

def get_model_predictions(model,test_data,prediction_candles):
    scaler = MinMaxScaler(feature_range=(0,1))
    model_inputs = test_data
    model_inputs = model_inputs.reshape(-1,1)
    model_inputs = scaler.fit_transform(model_inputs)
    x_test=[]
    for x in range(len(model_inputs)-prediction_candles-1):
        x_test.append(model_inputs[x:x+prediction_candles,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)
    return prediction_prices

def get_model_predictions_n_build(closes,prediction_candles,days_into_future,test_data):
    x = get_x_train(closes,prediction_candles,days_into_future)
    y=get_y_train(closes,prediction_candles,days_into_future)
    model= build_model(x,y)
    return get_model_predictions(model,test_data,prediction_candles)

def get_model_predictions_n_build_all_closes(all_closes,prediction_candles,days_into_future,test_data,x_train_per):
    x = get_x_train(all_closes[:len(all_closes)*x_train_per],prediction_candles,days_into_future)
    y = get_y_train(all_closes[:len(all_closes)*x_train_per],days_into_future)
    model= build_model(x,y)
    return get_model_predictions(model,test_data,prediction_candles)