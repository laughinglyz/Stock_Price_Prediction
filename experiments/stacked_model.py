from code import load_daily_stock_price
from code import preprocess_RC, HSI_Dataset
from code import run_model, HSI_lstm, HSI_gru
import ta
import pandas as pd
import numpy as np
import torch
import os
from sklearn.ensemble import *
from sklearn.metrics import mean_squared_error,accuracy_score

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

PATH = './trained_models/1a'
tech_indicators = ['volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'momentum_mfi',
       'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_nvi', 'volume_vwap',
       'volatility_atr', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'trend_macd',
       'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
       'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx',
       'trend_adx_pos', 'trend_adx_neg', 'trend_vortex_ind_pos',
       'trend_vortex_ind_neg', 'trend_vortex_ind_diff', 'trend_trix',
       'trend_mass_index', 'trend_cci', 'trend_dpo', 'trend_kst',
       'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
       'trend_psar_down', 'trend_psar_up_indicator',
       'trend_psar_down_indicator', 'momentum_rsi', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
       'momentum_ao', 'momentum_kama', 'momentum_roc', 'others_dr',
       'others_dlr', 'others_cr']

df = load_daily_stock_price()
df = df.dropna()
df = ta.add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", True)
data_X, data_Y, train_X, train_Y, valid_X, valid_Y, test_X, test_Y = preprocess_RC(df)
os.chdir(PATH)
vars = ["Open", "High", "Low", "Close"]+tech_indicators

train_features = train_X[:,:,0]
valid_features = valid_X[:,:,0]
test_features = test_X[:,:,0]
starts = test_X[:,0,0]
St = test_X[:,-1,0]
St = (St * starts + starts)
inv_test_Y = (test_Y * starts + starts)

ada = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
bagging = BaggingRegressor(n_estimators=500)
et = ExtraTreesRegressor(n_estimators=500)
gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1)
rf = RandomForestRegressor(n_estimators=500)

losses,RMSEs,MAPEs,accuracies = [],[],[],[]

plt.plot(range(test_Y.shape[0]),inv_test_Y,label="real price")
model_names = ["AdaBoost","Bagging","ExtraTree","GradientBoosting","RandomForest"]
for i,model in enumerate([ada,bagging,et,gb,rf]):

    model.fit(train_features,train_Y)
    prediciton = model.predict(test_features)
    loss = mean_squared_error(test_Y,prediciton)
    inv_pred = (prediciton * starts + starts)

    test_RMSE = np.sqrt(mean_squared_error(inv_pred,inv_test_Y))
    test_MAPE = np.mean(np.abs((inv_test_Y-inv_pred)/inv_test_Y)) * 100
    directional_true,directional_prediction = np.ones(test_X.shape[0]),np.ones(test_X.shape[0])
    directional_true[St>=inv_test_Y] = -1
    directional_prediction[St>=inv_pred] = -1
    test_accuracy = accuracy_score(directional_true, directional_prediction)
    
    losses.append(loss)
    RMSEs.append(test_RMSE)
    MAPEs.append(test_MAPE)
    accuracies.append(test_accuracy)

    plt.plot(range(test_Y.shape[0]),inv_pred,label=str(model_names[i])+" prediction")

plt.legend()
plt.show()

plt.bar(model_names,losses)
plt.title("loss")
plt.show()

plt.bar(model_names,RMSEs)
plt.title("RMSE")
plt.show()

plt.bar(model_names,MAPEs)
plt.title("MAPE")
plt.show()

plt.bar(model_names,accuracies)
plt.title("Accuracy")
plt.show()