# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 10:48:05 2021

Linke prototype code:
    https://towardsdatascience.com/anomaly-detection-for-dummies-15f148e559c1

Link clean_dataset function:
    https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
    
@author: 41162395
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib
from sklearn.ensemble import IsolationForest

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

ballshear = pd.read_csv("ballshear.csv")

df = ballshear[['SD','MEANX']]
clean_dataset(df)
df.head()

# plt.scatter(range(df.shape[0]), np.sort(df[X].values))
# plt.xlabel('index')
# plt.ylabel(X)
# plt.title(X + "distribution")
# sns.despine()

X = 'SD'

def outlier(X):
    sns.distplot(df[X])
    plt.title("Distribution of " + X)
    sns.despine()
    
    isolation_forest = IsolationForest(contamination=0.01) # "auto" or 0.0-0.5 outliner threshold
    isolation_forest.fit(df[X].values.reshape(-1, 1))
    
    xx = np.linspace(df[X].min(), df[X].max(), len(df)).reshape(-1,1)
    anomaly_score = isolation_forest.decision_function(xx)
    outlier = isolation_forest.predict(xx)
    plt.figure(figsize=(10,4))
    plt.plot(xx, anomaly_score, label='anomaly score')
    plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                     where=outlier==-1, color='r', 
                     alpha=.4, label='outlier region')
    plt.legend()
    plt.ylabel('anomaly score')
    plt.xlabel(X)
    plt.show();

for X in ('MEANX','SD'):
    outlier(X)