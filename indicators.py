import pandas as pd
import numpy as np
from ta.trend import ADXIndicator


def rsi(close, window):
    diff = close.diff(1)
    up = diff.where(diff > 0, 0.0)
    dn = -diff.where(diff < 0, 0.0)
    emaup = up.ewm(alpha=1 / window, min_periods=window).mean()
    emadn = dn.ewm(alpha=1 / window, min_periods=window).mean()
    rs = emaup / emadn
    rsi_ret = pd.Series(np.where(emadn == 0, 100, 100 - (100 / (1 + rs))), index=close.index)
    return rsi_ret


def stochastic(high, low, close, window_k, window_d, window_smooth):
    smin = low.rolling(window_k, min_periods=window_k).min()
    smax = high.rolling(window_k, min_periods=window_k).max()
    stoch_k = 100 * (close - smin) / (smax - smin)

    stoch_k = stoch_k.rolling(window_smooth, min_periods=window_smooth).mean()

    stoch_d = stoch_k.rolling(window_d, min_periods=window_d).mean()
    return stoch_k, stoch_d


def atr(high, low, close, window):
    cs = close.shift(1)
    tr1 = high - low
    tr2 = (high - cs).abs()
    tr3 = (low - cs).abs()
    tr = pd.DataFrame(data={'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    ave_tr = np.zeros(len(close))
    ave_tr[window - 1] = tr[0:window].mean()
    for i in range(window, len(ave_tr)):
        ave_tr[i] = (ave_tr[i - 1] * (window - 1) + tr.iloc[i]) / float(window)
    ave_tr = pd.Series(data=ave_tr, index=tr.index)
    return ave_tr


def adx(high, low, close, window):
    hd = high.diff(1)
    ld = -low.diff(1)
    df_temp = pd.DataFrame()
    df_temp["High"] = hd
    df_temp["Low"] = ld
    df_temp["dx+"] = np.where((df_temp["High"]>0.0) & (df_temp["High"]>df_temp["Low"]), df_temp["High"], 0.0)
    df_temp["dx-"] = np.where((df_temp["Low"]>0.0) & (df_temp["Low"]>df_temp["High"]), df_temp["Low"], 0.0)
    df_temp["smooth_dx+"] = df_temp["dx+"].rolling(window=window, min_periods=window).mean()
    df_temp["smooth_dx-"] = df_temp["dx-"].rolling(window=window, min_periods=window).mean()
    df_temp["ATR"] = atr(high, low, close, window)
    df_temp["DMI+"] = 100.0 * df_temp["smooth_dx+"] / df_temp["ATR"]
    df_temp["DMI-"] = 100.0 * df_temp["smooth_dx-"] / df_temp["ATR"]
    df_temp["DX"] = 100.0 * abs(df_temp["DMI+"] - df_temp["DMI-"]) / abs(df_temp["DMI+"] + df_temp["DMI-"])
    df_temp["ADX"] = df_temp["DX"].rolling(window=window, min_periods=window).mean()
    return df_temp
