# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:52:17 2017

@author: rreddy
"""
import datetime as dt
import numpy as np
import pandas as pd
import sklearn

from sklearn.ensemble import RandomForestClassifier
from pandas_datareader.data import DataReader
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.qda import QDA
from sklearn.svm import LinearSVC, SVC

def create_lagged_series(symbol, start_date, end_date, lags=5):
    """
    This creates a Pandas DataFrame that stores the percentage returns of 
    the adjusted closing value of a stock obtained from Yahoo Finance, 
    along with a number of lagged returns from the prior trading days 
    (defaults to 5 days).  Trading volume as well as the direction from 
    the previous day, are also     included.
    """
    
    ts = DataReader(symbol, 'yahoo', start_date - dt.timedelta(365), end_date)
    tslag = pd.DataFrame(index=ts.index)
    tslag['Today'] = ts['Adj Close']
    tslag['Volume'] = ts['Volume']
    
    for i in range(0, lags):
        tslag['Lag%s' % str(i+1)] = ts['Adj Close'].shift(i+1)
        
    tsret = pd.DataFrame (index= tslag.index)
    tsret['Volume'] = tslag['Volume']
    tsret['Today']  = tslag['Today'].pct_change()*100.0
    
    for i, x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret['Today'][i]=0.0001
                 
    for i in range(0, lags):
        tsret["Lag%s" %str(i+1)] = tslag['Lag%s' % str(i+1)].pct_change()*100.0
         
         
    tsret['Direction'] = np.sign(tsret['Today'])
    tsret = tsret[tsret.index >= start_date]
    
    return tsret


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    snpret = create_lagged_series("^GSPC", 
                                  dt.datetime(2010, 1, 1),
                                dt.datetime(2017, 5, 5))
    x = snpret[["Lag1","Lag2"]]
    y = snpret["Direction"]
    
    start_test = dt.datetime(2015,1,1)
    
    x_train = x[x.index < start_test]
    x_test  = x[x.index >= start_test]
    y_train = y[y.index < start_test]
    y_test  = y[y.index >= start_test]
    
    print("Hit Rates / Confusion Matrices: \n")
    models = [("LR", LogisticRegression()),
              ("LDA", LinearDiscriminantAnalysis()),
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(C=1000000.0, cache_size=200, class_weight=None,
                           coef0=0.0, degree=3, gamma=0.0001, kernel='rbf',
                           max_iter=-1, probability=False, random_state=None,
                           shrinking=True, tol=0.001, verbose=False)),
                ("RF", RandomForestClassifier(
                        n_estimators=1000, criterion='gini', max_depth=None,
                        min_samples_leaf=1, max_features='auto', 
                        bootstrap=True, oob_score=False, n_jobs=1, 
                        random_state=None, verbose=0))]
    
    for m in models:
        m[1].fit(x_train, y_train)
        pred = m[1].predict(x_test)
        print("%s:\n %0.3f" % (m[0], m[1].score(x_test, y_test)))
        print("%s\n" % confusion_matrix(pred, y_test))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    