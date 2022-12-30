from datetime import date
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

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
print(data.index)

data.plot.line(y="Close",use_index = True)
data["Tomorrow"] = data["Close"].shift(-1)


data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)


model = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)

train = data.iloc[:-100]
test = data.iloc[-100:]

predictors = ["Close", "Volume", "Open","High","Low"]
model_fitted_print =  model.fit(train[predictors],train["Target"])
print(model_fitted_print)

predictions = model.predict(test[predictors])
predictions = pd.Series(predictions,index=test.index)

print(precision_score(test["Target"],predictions))

combined = pd.concat([test["Target"],predictions],axis=1)

combined.plot()
plt.show()


#Util Functions
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions = backtest(data, model, predictors)

updatedScore = precision_score(predictions["Target"], predictions["Predictions"])

print("Second Score: " + str(updatedScore))

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = data.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    data[ratio_column] = data["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]


data = data.dropna(subset=data.columns[data.columns != "Tomorrow"])

modelFin = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


predictions_fin = backtest(data, model, new_predictors)
predictions_fin["Predictions"].value_counts()

print(precision_score(predictions_fin["Target"], predictions_fin["Predictions"]))

predictions_fin["Target"].value_counts() / predictions_fin.shape[0]

print(predictions_fin)












