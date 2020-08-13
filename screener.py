import yfinance as yf
from indicators import rsi, stochastic, adx
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def datetime_index(col):
    s = "%s %s" % (datetime.strftime(col["date"], "%Y-%m-%d"), col["minute"])
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def process_data(df):
    df["index"] = df.apply(datetime_index, axis=1)
    df.set_index("index", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def retrieve_data(ticker, param_dic):
    token = open(param_dic["token_path"]).readline()[:-1]
    today = datetime.today()
    yesterday = today - timedelta(days=1)
    if param_dic["time_frame"] == "1m" or param_dic["time_frame"] == "5m":
        url1 = f"https://cloud.iexapis.com/stable/stock/{ticker}/chart/date/" \
               f"{datetime.strftime(today, '%Y%m%d')}?token={token}"
        url2 = f"https://cloud.iexapis.com/stable/stock/{ticker}/chart/date/" \
               f"{datetime.strftime(yesterday, '%Y%m%d')}?token={token}"
        df = process_data(pd.concat([pd.read_json(url1), pd.read_json(url2)]))
        if param_dic["time_frame"] == "5m":
            df = df.resample('5T').agg({'open': 'first',
                                        'high': 'max',
                                        'low': 'min',
                                        'close': 'last',
                                        'volume': 'sum'})
        return df
    elif param_dic["time_frame"] == "1h" or param_dic["time_frame"] == "4h":
        url = f"https://cloud.iexapis.com/stable/stock/{ticker}/chart/1mm?token={token}"
        df = process_data(pd.read_json(url))
        df = df.resample(param_dic["time_frame"]).agg({'open': 'first',
                                                       'high': 'max',
                                                       'low': 'min',
                                                       'close': 'last',
                                                       'volume': 'sum'}).dropna()
        return df
    elif param_dic["time_frame"] == "1d":
        url = f"https://cloud.iexapis.com//stable/stock/{ticker}/chart/1y?token={token}"
        return process_data(pd.read_json(url))


def tick_process(ticker, param_dic):
    df = retrieve_data(ticker, param_dic)
    df["Volume_ave"] = df["Volume"].ewm(alpha=1 / param_dic["vol_window"], min_periods=param_dic["vol_window"]).mean()
    df["Volume_above_ave"] = df["Volume"] > df["Volume_ave"]
    df["RSI"] = rsi(df["Close"], param_dic["RSI_window"])
    df["RSI_ave"] = df["RSI"].ewm(alpha=1 / param_dic["RSI_ave"], min_periods=param_dic["RSI_ave"]).mean()
    rsi_conditions = [df["RSI"] > param_dic["RSI_hi"], df["RSI"] < param_dic["RSI_lo"]]
    rsi_choices = [1, -1]
    df["RSI_hilo"] = np.select(rsi_conditions, rsi_choices, default=0)
    df["Stoch_k"], df["Stoch_d"] = stochastic(df["High"], df["Low"], df["Close"], param_dic["Stochastic"])
    stoch_conditions = [df["Stoch_k"] > param_dic["Stoch_hi"], df["Stoch_k"] < param_dic["Stoch_lo"]]
    stoch_choices = [1, -1]
    df["Stoch_hilo"] = np.select(stoch_conditions, stoch_choices, default=0)
    df_temp = adx(df["High"], df["Low"], df["Close"], param_dic["ADX_window"])
    df = pd.concat([df, df_temp], axis=1)
    df = df[["Open", "High", "Low", "Close", "Volume", "Volume_above_ave",
             "RSI", "RSI_ave", "RSI_hilo",
             "Stoch_k", "Stoch_d", "Stoch_hilo",
             "DMI+", "DMI-", "Cross", "ADX"]]
    s = df.loc[df.index[-1]]
    s.name = ticker
    return s


def screener(tick_path, param_dic, limit=1000):
    tickers = pd.read_excel(tick_path)["Ticker"]
    result_ls = []
    length = len(tickers)
    counter = 1
    for t in tickers:
        print(f"{counter}/{length}: {t}")
        result_ls.append(tick_process(t, param_dic))
        if counter == limit:
            break
        counter += 1
    df = pd.concat(result_ls, axis=1).swapaxes(axis1=0, axis2=1)
    return df
