import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

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

def preprocess(df, period = 20):
    df['Mid'] = df[['High','Low']].mean(axis = 1)
    df = df[['Mid', "Open", "High", "Low", "Close"]+tech_indicators].copy(deep=True)
    scaler =  MinMaxScaler()
    data = scaler.fit_transform(df)
    scaler.fit(df['Mid'].to_numpy().reshape(-1,1))
    data_X, data_Y = [],[]
    for i in range(data.shape[0]-period-5):
        data_X.append(data[i:i+period])
        data_Y.append(df.iloc[i+period+5]['Mid'])
    data_X, data_Y = np.array(data_X), np.array(data_Y)
    training_size = int(np.round(0.6*data.shape[0]))
    valid_size = int(np.round(0.2*data.shape[0]))
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = \
        data_X[:training_size,:], data_X[training_size+period:training_size+valid_size+period,:], data_X[training_size+valid_size+period:,:],\
        data_Y[:training_size], data_Y[training_size+period:training_size+valid_size+period], data_Y[training_size+valid_size+period:]
    # scaler.fit(train_Y.reshape(-1,1))
    return data_X, data_Y, train_X, scaler.transform(train_Y.reshape(-1,1)), valid_X, valid_Y, test_X, test_Y, scaler 

def preprocess_RC(df, period = 20):
    df['Mid'] = df[['High','Low']].mean(axis = 1)
    df = df[['Mid', "Open", "High", "Low", "Close"]+tech_indicators].copy(deep=True)
    scaler =  MinMaxScaler()
    df.loc[:,["Open", "High", "Low", "Close"]+tech_indicators] = scaler.fit_transform(df[["Open", "High", "Low", "Close"]+tech_indicators])
    data_X, data_Y = [],[]
    for i in range(df.shape[0]-period-5):
        start = df.iloc[i]['Mid']
        x, y = df.iloc[i:i+period].to_numpy(), df.iloc[i+period+5]['Mid']
        x, y = (x-start)/start, (y-start)/start
        x[0,0] = start
        data_X.append(x)
        data_Y.append(y)
    data_X, data_Y = np.array(data_X), np.array(data_Y)
    training_size = int(np.round(0.6*df.shape[0]))
    valid_size = int(np.round(0.2*df.shape[0]))
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = \
        data_X[:training_size,:], data_X[training_size+period:training_size+valid_size+period,:], data_X[training_size+valid_size+period:,:],\
        data_Y[:training_size], data_Y[training_size+period:training_size+valid_size+period], data_Y[training_size+valid_size+period:]
    return data_X, data_Y, train_X, train_Y, valid_X, valid_Y, test_X, test_Y
