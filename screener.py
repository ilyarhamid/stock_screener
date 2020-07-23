import yfinance as yf
from indicators import rsi, stochastic, adx
import numpy as np
import pandas as pd


def tick_process(ticker, param_dic):
    t = yf.Ticker(ticker)
    df = t.history("200d")
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
