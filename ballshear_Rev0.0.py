# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 21:14:17 2021
Wed Sep 29 3:48PM : Improve comment

Link :
    Isolation forest model : https://github.com/mesmalif/Practical_Machine_learning/tree/develop_practical_ML
    seaborn : https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed


@author: 41162395
"""
import numpy as np
import pandas as pd
import seaborn as sns
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
#from sklearn.preprocessing import MinMaxScaler

#Import raw dataset csv file
df_org = pd.read_csv('ballshear.csv')
df = pd.read_csv('ballshear.csv')

#Data cleasing
to_drop = ['LSL','USL','Parameter.Recipe','PROJECT_TYPE']
df_org.drop(to_drop, inplace=True, axis=1)
df_org.dropna(inplace=True)
df.drop(to_drop, inplace=True, axis=1)
df.dropna(inplace=True)
df_org.dropna(inplace=False)

#Drop column features which are not relative anomaly
to_drop = ['DATE_TIME','C_RISTIC','CUSTOMER','PT','EN_NO','DEVICE','REMARK',
            'SUBGRP','BOM_NO','MC_ID','MC_NO','COUNTER',
            'CHAR_MINOR','PACKAGE','DATE_','CIMprofile.cim_machine_name',
            'Parameter.DataType','Parameter.Unit',
            'Parameter.Valid','Parameter.EquipOpn','Parameter.EquipID',
            'Parameter.ULotID','Parameter.CreateTime']
df.drop(to_drop, inplace=True, axis=1)

#For json input for testing prediction (I get its from index0 dfx = df[:1])
dfx = pd.read_json('x_input.json')

# One hot encoder to numpy array X as below 7 columns or features
# SHIFT, WIRE_SIZE, PLANT_ID, Parameter.Group, Parameter.No, Parameter.BondType, Parameter.No_1  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,3,5,6,7,8])], remainder='passthrough')
X = np.array(ct.fit_transform(df))
X0 = np.array(ct.transform(dfx))

# Fit Anomality model
clf = IsolationForest(contamination=0.001) # "auto" or 0.0-0.5 outliner threshold
clf.fit(X)

# numpy array Y and SCORE for anomallity predictions.
Y = clf.predict(X)
SCORE = clf.score_samples(X)
Y0 = clf.predict(X0)
SCORE0 = clf.score_samples(X0)

#add 'Y' and 'SCORE' array as new column in  df DataFrame
df_org['Y'] = Y.tolist()
df_org['SCORE'] = SCORE.tolist()
dfx['Y'] = Y0.tolist()
dfx['SCORE'] = SCORE0.tolist()

#Export output to csv
df_org.to_csv('ballshear_anomality.csv')

sns.pairplot(df_org[["MEANX","SD","SCORE","Y"]])

#############################
# Common command 
#-------------------------
# df.describe
# df.info()
# df.index
# df.columns
# dfx[:5]
# du1=ds1.unstack()
# df.isna().sum()
# dfx = df.set_index(['C_RISTIC','PT','Parameter.CreateTime','Parameter.BondType','Parameter.No'])
# df.loc[25024]
# for i in range (len(Y)):
#     if Y[i] == -1:
#         print (df.loc[i])
#
# selected_columns = df[["col1","col2"]]
# new_df = selected_columns.copy()

# df_org[["MEANX","SD","Y"]]

# sns.scatterplot(x='sepal_length', y='sepal_width', hue='class', data=iris)
# sns.scatterplot(x='MEANX', y='SD', data=temp)
#
# df0 = pd.read_json(jsonstr,orient="index")
# df0 = pd.read_json('input.json',orient="index")
# df0.drop(to_drop, inplace=True)
# df0 = df0.transpose()
# 
# df_org[Y==-1]  #List Row for Y = -1 
# df0json = df0.to_json()
# file1 = open("x_input.json","w")
# file1.writelines(df0json)
# file1.close() 

