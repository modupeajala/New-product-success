# -*- coding: utf-8 -*-
"""

@author: MODUPE AJALA
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import KFold
import seaborn as sns
from sklearn.model_selection import train_test_split
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.helmert import HelmertEncoder
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from category_encoders.sum_coding import SumEncoder
from statsmodels.tsa.api import VAR
from category_encoders.one_hot import OneHotEncoder

from sklearn.preprocessing import StandardScaler
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow



#%matplotlib inline
#import mpld3



path_crewneck = r'C:\Users\Modupe Ajala\Desktop\CACC Cleaned Data\Crewneck.csv'
crewneck = pd.read_csv(path_crewneck) 

crewneck = crewneck.drop('DATE',1)
crewneck = crewneck.drop('PRODUCT_CODE',1)
#crewneck = crewneck.drop('CHANNEL',1)
crewneck = crewneck.drop('STARTING_INV',1)
#crewneck = crewneck.drop('STYLE',1)
#crewneck = crewneck.drop('COLOR',1)
crewneck = crewneck.drop('ATTRIBUTION',1)
crewneck = crewneck.drop('FISCAL_YEAR',1)
crewneck = crewneck.drop('FISCAL_MONTH',1)
crewneck = crewneck.drop('FISCAL_WEEK',1)
crewneck = crewneck.drop('STRATEGIC_BUSINESS_UNIT_DESC',1)
crewneck = crewneck.drop('LEVEL_OF_ENGAGEMENT',1)
crewneck = crewneck.drop('STYLE_GROUP',1)
#crewneck = crewneck.drop('INVENTORY_GROUP',1)
crewneck = crewneck.drop('INVTY_CAP_GROUP',1)
crewneck = crewneck.drop('CAPACITY_GROUP',1)
crewneck = crewneck.drop('BRAND_NAME',1)
crewneck = crewneck.drop('GENDER_DESC',1)
#crewneck = crewneck.drop('GENDER_CATEGORY_DESC',1)
crewneck = crewneck.drop('CATEGORY_DESC',1)
crewneck = crewneck.drop('SUBCATEGORY_DESC',1)
#crewneck = crewneck.drop('FABRICATION',1)
crewneck = crewneck.drop('SILHOUETTE',1)



DATE = crewneck['YEAR'].map(str) + '-' + crewneck['MONTH'].map(str) + '-' + crewneck['DAY'].map(str)
DATE = DATE.to_frame()
df = pd.concat([crewneck, DATE], axis=1)
df = df.rename(columns={0: 'DATE'})

#g = sns.relplot(x="DATE", y="DEMAND", kind="line",height=5, aspect=10, data=df)   # Demand Plot


df.DATE = pd.DatetimeIndex(df.DATE).to_period('W')
df.set_index('DATE', inplace = True)
df = df.sort_values(by=['DATE'])

df = df.drop('DAY',1)
df = df.drop('MONTH',1)
df = df.drop('YEAR',1)


# Splitting Data

X_train, X_test , y_train, y_test = train_test_split(df.loc[:, df.columns != 'DEMAND'], df['DEMAND'], test_size=0.33, random_state=42)


df =df.values
tscv = TimeSeriesSplit(n_splits=5)
print(tscv)
for train_index, test_index in tscv.split(df):
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = df[train_index], df[test_index]
     y_train, y_test = df[train_index], df[test_index]



train = df.loc[:, df.columns != 'DEMAND']
test = df['DEMAND']

# Scale Data


encoder = SumEncoder(cols=['FABRICATION'])   

encoder = SumEncoder(cols=['CHANNEL', 'STYLE', 'COLOR', 'INVENTORY_GROUP',    
                               'GENDER_CATEGORY_DESC', 'FABRICATION', 'SILHOUETTE']) 
    
encoder = OneHotEncoder(cols=['FABRICATION'])   

encoder = OneHotEncoder(cols=['CHANNEL', 'STYLE', 'COLOR', 'INVENTORY_GROUP',    
                               'GENDER_CATEGORY_DESC', 'FABRICATION']) 

train = encoder.fit_transform(train)

train = train.drop(['intercept'], axis=1)

test = test.values
test = test.reshape(-1,1)
scaler = StandardScaler()
scaler.fit(test)
test = scaler.transform(test)
test = pd.DataFrame(test)

# Encoder

encoder = SumEncoder(cols=['FABRICATION'])   

encoder = SumEncoder(cols=['CHANNEL', 'STYLE', 'COLOR', 'INVENTORY_GROUP',    
                               'GENDER_CATEGORY_DESC', 'FABRICATION', 'SILHOUETTE']) 

df = encoder.fit_transform(df)

df = df.drop(['intercept'], axis=1)


# Fitting Model
tensorflow.random.set_seed(42)



train = train['CHANNEL_1'].astype(float)

n_cols = train.shape[1]
n_features = train.shape[0]
generator = TimeseriesGenerator(train, test, length=n_cols, batch_size=6)


model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_cols,n_features)))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator,epochs=90)




# Proves Data is Stationary

X = df.DEMAND
split = round(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print('mean1=%f, mean2=%f' % (mean1, mean2))
print('variance1=%f, variance2=%f' % (var1, var2))


df['DEMAND'].hist(bins = 10, range =(0, 200)) 


X = df['DEMAND']
result = adfuller(X) 
print('ADF Statistic: %f' % result[0])
print('p-value: %.20f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))





#LSTM











# VAR Model --> https://www.statsmodels.org/stable/vector_ar.html?highlight=var#module-statsmodels.tsa.vector_ar.var_model

df =df.values

model = VAR(df)

results = model.fit(2)

results.summary()

results.plot()

results.month_plot()



results.plot_acorr()

model.select_order(15)

results = model.fit(maxlags=15, ic='aic')

lag_order = results.k_ar

results.forecast(data.values[-lag_order:], 5)

mpld3.enable_notebook(results.plot_forecast(13))

