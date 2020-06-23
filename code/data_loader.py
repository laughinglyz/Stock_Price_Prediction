import pandas as pd

def load_daily_stock_price():
    df = pd.read_csv('./data/^HSI.csv')
    return df

def load_weekly_stock_price():
    df = pd.read_csv('./data/^HSI_weekly.csv')
    return df