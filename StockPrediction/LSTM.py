import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential, load_model
import matplotlib.pyplot as plt
from datetime import date

#Use Moon Environment

plt.style.use("fivethirtyeight")

import yfinance as yf

START = "1990-01-01"
currentTime = date.today().strftime("%Y-%m-%d")
next_day =  pd.to_datetime(currentTime) + pd.DateOffset(days = 1)



#LOAD STOCK DATA
def load_data(ticker):
    data = yf.download(ticker,START,next_day)
    print(data)
    return data

# selected_stock = input("Type Ticker Symbol: ")
selected_stock = "AAPL"

data = load_data(selected_stock)
data = pd.DataFrame(data)
print(data)