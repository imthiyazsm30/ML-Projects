# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 20:14:27 2020

@author: Chandramouli
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1:2]

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(training_set)

x_train=[]
y_train=[]
for i in range(60,1258):
    x_train.append(training_set_scaled[i-60:i,0])#column 0#stock price of (0-59)
    y_train.append(training_set_scaled[i,0])#stock price of t+1(60)
x_train,y_train=np.array(x_train),np.array(y_train)
    
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

#importing the keras libraries and packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#initializing the RNN
regressor=Sequential()

#adding the first LSTM layer and some Dropout regularization to avoid overfitting
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))#dropping 20% neurons in lstm layer during training unwanted data

#adding second lstm layer and drop out regulari
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#adding 3rd lstm layer and dropout regulari
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

#adding 4th lstm layer and dropout regulari
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#adding the op layer
regressor.add(Dense(units=1))

#compiling the rnn
regressor.compile(optimizer='adam',loss='mean_squared_error')

#fitting the RNN to the training set
regressor.fit(x_train,y_train,batch_size=32,epochs=100)

dataset_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=dataset_test.iloc[:,1:2]

dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)#vertical axis=0
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].value

inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])#column 0#stock price of (0-59)
x_test=np.array(x_test)


#reshape to 3d for rnn ip
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predicted_stock_price=regressor.predict(x_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)

#visualizing the results
plt.plot(real_stock_price,color='red',label='Real google stock price')
plt.plot(predicted_stock_price,color='blue',label='predicted google stock price')
plt.title("google stock price prediction")
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()


import math
from sklearn.metrics import mean_squared_error
rmse=math.sqrt(mean_squared_error(real_stock_price,predicted_stock_price))