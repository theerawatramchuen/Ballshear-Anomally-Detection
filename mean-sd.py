# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 13:47:59 2021

Link prototype code:
    https://github.com/mesmalif/Practical_Machine_learning/tree/develop_practical_ML

Link clean_dataset function:
    https://stackoverflow.com/questions/31323499/sklearn-error-valueerror-input-contains-nan-infinity-or-a-value-too-large-for
    
@author: 41162395
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

ballshear = pd.read_csv('ballshear.csv')

df = ballshear[['SD','MEANX']]
clean_dataset(df)
df.head()

x = df.values

plt.scatter(x[:,0], x[:,1])

df.describe

clf = IsolationForest(contamination=0.01) # "auto" or 0.0-0.5 outliner threshold
clf.fit(x)
predictions = clf.predict(x)

(predictions<0).mean()

abn_ind = np.where(predictions < 0)

plt.scatter(x[:,0], x[:,1])
#plt.scatter(x[abn_ind,0], x[abn_ind,1], edgecolors="r")
plt.scatter(x[abn_ind,0], x[abn_ind,1])

plt.ylabel('MEANX')
plt.xlabel('SD')
plt.show();