import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PRICE =  ['Open', 'High', 'Low', 'Close']

def preprocess(df, period = 20):
    scaler =  MinMaxScaler()
    df2 = df[PRICE].copy()
    data = scaler.fit_transform(df)
    data_X, data_Y, mid_points = [],[],[]
    for i in range(data.shape[0]-period):
        data_X.append(data[i:i+period])
        data_Y.append(df2.iloc[i+period][['High','Low']].mean())
        mid_points.append(df2.iloc[i+period-1][['High','Low']].mean())
    data_X, data_Y, mid_points = np.array(data_X), np.array(data_Y), np.array(mid_points)
    train_X, test_X, train_Y, test_Y, train_m, test_m = train_test_split(data_X, data_Y, mid_points, train_size = 0.8, random_state = 3)
    train_X, valid_X, train_Y, valid_Y, train_m, valid_m = train_test_split(train_X, train_Y, train_m, train_size = 0.75, random_state = 3)
    scaler.fit(train_Y.reshape(-1,1))
    return train_X, scaler.transform(train_Y.reshape(-1,1)), valid_X, valid_Y, valid_m, test_X, test_Y, test_m, scaler