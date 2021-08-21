# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 21:14:17 2021

Link :
    Isolation forest model : https://github.com/mesmalif/Practical_Machine_learning/tree/develop_practical_ML
    seaborn : https://towardsdatascience.com/introduction-to-data-visualization-in-python-89a54c97fbed
    Dummy column for catagory feature : https://stackoverflow.com/questions/44601533/how-to-use-onehotencoder-for-multiple-columns-and-automatically-drop-first-dummy/44601764


@author: 41162395
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

df_org = pd.read_csv('wirepull.csv')
df = pd.read_csv('wirepull.csv')

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

# X as below 11 columns/features
# SHIFT, WIRE_SIZE, MEANX, PLANT_ID, Parameter.No, Parameter.BondType, Parameter.No_1  
df.drop(to_drop, inplace=True, axis=1)

# Convert categorical variable into dummy/indicator variables.
X2 = pd.get_dummies(df,drop_first=True)

# Fit Anomality model as clf class
clf = IsolationForest(contamination=0.003) # "auto" or 0.0-0.5 outliner threshold
clf.fit(X2)

# numpy array Y and SCORE for anomallity predictions.
Y = clf.predict(X2)
SCORE = clf.score_samples(X2)

#add 'Y' and 'SCORE' array as new column in  df DataFrame
df_org['Y'] = Y.tolist()
df_org['SCORE'] = SCORE.tolist()

#df1.to_excel("output.xlsx") 
df_org.to_csv('wirepull_anomality_v2.csv')

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