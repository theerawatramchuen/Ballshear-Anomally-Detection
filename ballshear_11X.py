# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 21:14:17 2021

Link :
    Isolation forest model : https://github.com/mesmalif/Practical_Machine_learning/tree/develop_practical_ML
    seaborn : https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed


@author: 41162395
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

df_org = pd.read_csv('ballshear.csv')
df = pd.read_csv('ballshear.csv')

to_drop = ['LSL','USL','Parameter.Recipe','PROJECT_TYPE']
df_org.drop(to_drop, inplace=True, axis=1)
df_org.dropna(inplace=True)
df.drop(to_drop, inplace=True, axis=1)
df.dropna(inplace=True)
#df = df_org

df_org.dropna(inplace=False)
#df = df_org

to_drop = ['DATE_TIME','C_RISTIC','CUSTOMER','PT','EN_NO','DEVICE','REMARK',
            'SUBGRP','BOM_NO','MC_ID','MC_NO','COUNTER',
            'CHAR_MINOR','PACKAGE','DATE_','CIMprofile.cim_machine_name',
            'Parameter.Group','Parameter.DataType','Parameter.Unit',
            'Parameter.Valid','Parameter.EquipOpn','Parameter.EquipID',
            'Parameter.ULotID','Parameter.CreateTime']

df.drop(to_drop, inplace=True, axis=1)


# One hot encoder to numpy array X as below 11 columns or features
# SHIFT, WIRE_SIZE, MEANX, PLANT_ID, Parameter.No, Parameter.BondType, Parameter.No_1  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,3,5,6,7])], remainder='passthrough')
X = np.array(ct.fit_transform(df))

# Fit Anomality model
clf = IsolationForest(contamination=0.003) # "auto" or 0.0-0.5 outliner threshold
clf.fit(X)

# numpy array Y and SCORE for anomallity predictions.
Y = clf.predict(X)
SCORE = clf.score_samples(X)

#add 'Y' and 'SCORE' array as new column in  df DataFrame
df_org['Y'] = Y.tolist()
df_org['SCORE'] = SCORE.tolist()

#df1.to_excel("output.xlsx") 
df_org.to_csv('ballshear_anomality.csv')

sns.pairplot(df)
#############################
# Common dataframe command 
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