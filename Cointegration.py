"""
Successful ALgorithmic Trading.
Chapter 10
"""

import datetime  as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
import pandas as pd
import pprint
import statsmodels.tsa.stattools as ts

from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
from pandas_datareader import data, yahoo
from statsmodels.api import OLS


def hurst(ts):
    lags = range(2,100)
    tau = [sqrt(std(subtract(ts[lag:],ts[:-lag]))) for lag in lags]
    poly = polyfit(log(lags), log(tau), 1)
    return 2*poly[0]

gbm = log(cumsum(randn(100000))+1000)
mr  = log(randn(100000)+1000)
tr  = log(cumsum(randn(100000)+1)+1000)


def plot_price_series(df, ts1, ts2):
    months = md.MonthLocator()
    fig, ax = plt.subplots()
    ax.plot(df.index, df[ts1], label=ts1)
    ax.plot(df.index, df[ts2], label=ts2)
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %y'))
    #ax.set_xLim(dt.datetime(2015,1,1), dt.datetime(2017,5,5))
    ax.grid(True)
    fig.autofmt_xdate()
    
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title('%s and %s Daily Prices' % (ts1, ts2))
    plt.legend()
    plt.show()
    
def plot_scatter_series(df, ts1, ts2):
    plt.xlabel('%s Price ($)' % ts1)
    plt.ylabel('%s Price ($)' % ts2)
    plt.title('%s and %s Price Scatter Plot' % (ts1, ts2))
    plt.scatter(df[ts1], df[ts2])
    plt.show()
    
def plot_residuals(df):
    months = md.MonthLocator()
    fig, ax = plt.subplots()
    ax.plot (df.index, df['res'], label='Residuals')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(md.DateFormatter('%b %Y'))
    ax.set_xlim(dt.datetime(2015,1,1), dt.datetime(2017,5,5))
    ax.grid(True)
    fig.autofmt_xdate()
    
    plt.xlabel('Month/Year')
    plt.ylabel('Price ($)')
    plt.title ('Residual Plot')
    plt.legend()
    plt.plot(df['res'])
    plt.show()
    
if __name__ == "__main__":
    start = dt.datetime(2012, 1, 2)
    end   = dt.datetime(2017, 5, 5)
    df = data.DataReader(['arex', 'wll'], "yahoo", start, end)
    df = df['Adj Close']
    
    plot_price_series(df, 'arex', 'wll')
    plot_scatter_series(df, 'arex', 'wll')
    model = OLS(df['wll'], df['arex'])
    results=model.fit()
    df['pred']=results.predict(df['arex'])
    df['res']=df['pred']-df['wll']
    plot_residuals(df)
    
    cadf = ts.adfuller(df['res'])
    pprint.pprint(cadf)
    
