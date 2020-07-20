import pandas as pd
import numpy as np


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
    return stoch_k, stoch_d  # This is a mock test
