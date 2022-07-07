import numpy as np
import pandas as pd
import pandas_profiling
from pandas_datareader import data as pdr
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import math
import statistics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import yfinance as yf
import datetime


def marketcap(df):

    unit = 1000000000
    df['MarktCap'] = df['Open'] * df['Volume'] / unit

    return df
