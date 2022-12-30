#Uses Base environment
import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import math
from sklearn.metrics import mean_squared_error

### Create the Stacked LSTM model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM

# from tensorflow import keras
# from keras.layers import Dense, LSTM
# from keras.models import Sequential, load_model
# import keras as kf

#Use Python 3.90

import numpy as np


START = "2010-01-01"
currentTime = date.today().strftime("%Y-%m-%d")
next_day =  pd.to_datetime(currentTime) + pd.DateOffset(days = 1)
print("NEXT DAY:" + str(next_day))
TEST = "2022-12-28"


st.title("Stock Prediction Application")

#Find a way for the user to import the stock name
stocks = ("AAPL", "GOOG","MSFT", "GME")
selected_stock = st.selectbox("Select dataset for predicition", stocks)
# n_years = st.slider("Years of prediction:", 1,4)
# period = n_years * 365


n_months = st.slider("Months of prediction:", 1,4)
period = n_months * 365



#LOAD STOCK DATA
def load_data(ticker):
    data = yf.download(ticker,START,next_day)
    print(data)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load Data...")
data = load_data(selected_stock)
print(data)


data_load_state.text("Loading data... done!")

#View raw data
st.subheader("Raw Data")
st.write(data.tail())

##########################LSTM MODEL
# #Predicted Stocks
# df1=data.reset_index()['Close']

# #LSTM Calculation
# scaler=MinMaxScaler(feature_range=(0,1))
# df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

# training_size=int(len(df1)*0.70)
# test_size=len(df1)-training_size
# train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# def create_dataset(dataset, time_step=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-time_step-1):
# 		a = dataset[i:(i+time_step), 0]  
# 		dataX.append(a)
# 		dataY.append(dataset[i + time_step, 0])
# 	return np.array(dataX), np.array(dataY)


# # reshape Train and Test Data
# time_step = 100
# X_train, y_train = create_dataset(train_data, time_step)
# X_test, ytest = create_dataset(test_data, time_step)

# # reshape input to be [samples, time steps, features] which is required for LSTM
# X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
# X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# model=Sequential()
# model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
# model.add(LSTM(50,return_sequences=True))
# model.add(LSTM(50))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error',optimizer='adam')

# model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)


# ### Lets Do the prediction and check performance metrics
# train_predict=model.predict(X_train)
# test_predict=model.predict(X_test)



# ##Transformback to original form
# train_predict=scaler.inverse_transform(train_predict)
# test_predict=scaler.inverse_transform(test_predict)


# ### Calculate RMSE performance metrics
# math.sqrt(mean_squared_error(y_train,train_predict))


# ### Test Data RMSE
# math.sqrt(mean_squared_error(ytest,test_predict))

# ### Plotting 
# # shift train predictions for plotting
# look_back=100
# trainPredictPlot = np.empty_like(df1)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(df1)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# # plot baseline and predictions
# plt.plot(scaler.inverse_transform(df1))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)















############################################################### Prophet Model

#Plot the raw data

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Open'],name = 'stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y = data["Close"],name = "stock_close"))
    fig.layout.update(title_text = "Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
df_train = data[['Date','Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

m = Prophet(daily_seasonality = True)
m.fit(df_train)

future = m.make_future_dataframe(periods = period)
forecast = m.predict(future)



#View raw data
st.subheader("Future Data")
st.write(forecast.tail())


#plot the data

fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)

st.write(fig2)






