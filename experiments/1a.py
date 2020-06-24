from code import load_daily_stock_price
from code import preprocess, HSI_Dataset
from code import run_model, HSI_lstm
import ta
import pandas as pd
import numpy as np
import torch
import os

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
data_X, data_Y, train_X, train_Y, valid_X, valid_Y, test_X, test_Y, scaler = preprocess(df)
os.chdir(PATH)
vars = ["Open", "High", "Low", "Close"]+tech_indicators
lstm_RMSE, lstm_MAPE, lstm_accuracy = [], [], []
gru_RMSE, gru_MAPE, gru_accuracy = [], [], []
for idx, indicator in enumerate(vars):
    train_features = train_X[:,:,[0, idx+10]]
    valid_features = valid_X[:,:,[0, idx+10]]
    test_features = test_X[:,:,[0, idx+10]]

    train_set = HSI_Dataset(train_features,train_Y)
    model = HSI_lstm(
        input_size=2,
        hidden_size=64,
        num_layers=1
    )

    model, train_loss, valid_loss, valid_RMSE, valid_MAPE, valid_accuracy, n_epochs = \
        run_model(model.float(), scaler, train_set=train_set, valid_X=valid_features, valid_Y=valid_Y, n_epochs=80, shuffle = False)

    plt.xlabel("epoch")
    plt.ylabel("training_loss")
    plt.plot(range(n_epochs),train_loss)
    plt.show()

    plt.xlabel("epoch")
    plt.ylabel("validation_loss")
    plt.plot(range(n_epochs),valid_loss)
    plt.show()

    plt.xlabel("epoch")
    plt.ylabel("validation_accuracy")
    plt.plot(range(n_epochs),valid_accuracy)
    plt.show()

    loss, test_RMSE, test_MAPE, test_accuracy = \
        run_model(model, scaler, running_mode='test', test_X=test_features, test_Y=test_Y)

    print(loss)
    print(test_RMSE)
    print(test_MAPE)
    print(test_accuracy)
    lstm_RMSE.append(test_RMSE)
    lstm_MAPE.append(test_MAPE)
    lstm_accuracy.append(test_accuracy)

    # filename = indicator+".pt"
    # model.load_state_dict(torch.load(filename))
    # model.eval()

    inputs = torch.from_numpy(test_features)
    inputs.to(torch.device('cpu'))
    with torch.no_grad():
        outputs = scaler.inverse_transform(model(inputs).numpy().reshape(-1,1)).reshape(-1,)
        plt.xlabel("time")
        plt.ylabel("HSI value")
        plt.plot(range(len(outputs)),outputs,label="prediction")
        plt.plot(range(len(outputs)),test_Y,label="actual")
        plt.legend()
        plt.show()

    filename = indicator+".pt"
    f = open(filename,"w+")
    f.close()
    torch.save(model.state_dict(), filename)
    break
 
# plt.xticks(range(len(vars)),vars)