# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 10:05:43 2022
[[187862]
 [127095]
 [170287]
 [197733]]
[199472.20002557 133730.09349061 180458.30231214 210151.35702865]
@author: Tejas
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model



def get_data_from_excel():
    l_df=pd.read_excel(
       io='Book.xlsx',
       engine='openpyxl',
       sheet_name='Sheet1',
       usecols='A:Q',
       nrows=24,
    )
    return l_df
df =get_data_from_excel()
df.head()
df.tail()
df.shape

df.isnull().sum()
df.info()

df['Date'].str.replace("-","").astype(str)

df.head()
df_prod = df.groupby(by='Date').mean()
df_prod
# Total Production_df.info()
df['Total sales'].unique()
d_dummies = pd.get_dummies(df['Date'], drop_first = True)
d_dummies.head()
df = pd.concat([df, d_dummies], axis = 1)
df.head()
df.describe()
df.drop(['Date'], axis = 1, inplace = True)
df.head()
# df_region = Total Production_df.groupby(by='').mean()
# df_region
# Total Production_df.columns
# Total Production_df[['Total Total Production',  'Domestic Total sales Total sales',  'Total Total Exported salesed sales', 'Total sales']].hist(bins = 30, figsize = 20,20), color = 'r')
df[['Total Production', 'Domestic Total sales','Total Exported sales','Total sales']].hist(bins = 10, figsize = (20,20), color = 'r');
# plot pairplot
sns.pairplot(df)
# Regression Plot (No Machine Learning)
sns.regplot(x = 'Total Production', y = 'Domestic Total sales', data = df)
plt.show()
# Regression Plot (No Machine Learning)
sns.regplot(x = 'Total Production', y = 'Total Exported sales', data = df)
plt.show()
# Check Correlation
corr = df.corr()
# Heatmap for Correlation
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True)
plt.show()
#"part training"

df.columns
X = df.drop(columns =['Date'])
y = df['Date']
# Check X
X.head()
# Check y
y.head()
# Check Shape
X.shape
# Check Shape
y.shape
# Casting to NP Arrays
X = np.array(X)
y = np.array(y)
# Reshaping of y
y = y.reshape(-1,1)
y.shape
#Scaling the data numerical data before feeding the model
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
X = scaler_x.fit_transform(X)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)
X
y
# Split the data into 20% Testing and 80% Training
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=80)
# Shape Checking
X_train.shape
# Shape Checking
X_test.shape
#"part 5 sk learn"

# Using Linear Regression Model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Get the predictions
y_predict = regressor.predict(X_test)
y_predict.shape

# Get the Values "before" scaling
y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)
# Number of Features and Cases
k = X_test.shape[1] # Number of Features
n = len(X_test) # Number of Cases
print("Features:",k)
print("Cases:",n)
# Metrics Calculation

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE=mean_absolute_error(y_test_orig, y_predict_orig)
r2=r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1 - (1 - r2) * (n -1) / (n - k -1)
# Evaluation Results Printing
print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

# Columns Check
df.columns
# Check the Weights for the various Features
list(zip(['Total Production', 'Domestic Total sales','Total Exported sales','Total sales'], regressor.coef_[0])) 

#main features

X_3f = df[['Total Production']].values
Y_3f = df['Total sales'].values
# Casting to NP Arrays
X_3f = np.array(X_3f)
Y_3f = np.array(Y_3f)
# Reshaping of y
Y_3f = Y_3f.reshape(-1,1)
#Scaling the data numerical data before feeding the model
#from sklearn.preprocessing import MinMaxScaler

scaler_x3f = MinMaxScaler()
X_3f = scaler_x3f.fit_transform(X_3f)

scaler_y3f = MinMaxScaler()
Y_3f = scaler_y3f.fit_transform(Y_3f)
# Split the data into 20% Testing and 80% Training
#from sklearn.model_selection import train_test_split

X3f_train,X3f_test,y3f_train,y3f_test = train_test_split(X_3f,Y_3f,test_size=0.20,random_state=0)
# Using Linear Regression Model
#from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X3f_train, y3f_train)
# Get the predictions
y3f_predict = regressor.predict(X3f_test)
print(y3f_predict)
# Get the Values "before" scaling
y3f_predict_orig = scaler_y3f.inverse_transform(y3f_predict)
y3f_test_orig = scaler_y3f.inverse_transform(y3f_test)
# Number of Features and Cases
k = X3f_test.shape[1] # Number of Features
n = len(X3f_test) # Number of Cases
print("Features:",k)
print("Cases:",n)
# Metrics Calculation
#from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RMSE = float(format(np.sqrt(mean_squared_error(y3f_test_orig, y3f_predict_orig)),'.3f'))
MSE = mean_squared_error(y3f_test_orig, y3f_predict_orig)
MAE=mean_absolute_error(y3f_test_orig, y3f_predict_orig)
r2=r2_score(y3f_test_orig, y3f_predict_orig)
adj_r2 = 1 - (1 - r2) * (n -1) / (n - k -1)
# Evaluation Results Printing
print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

#tensor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
import joblib

# Deep Neural Network
model = keras.Sequential()
model.add(Dense(32, input_dim = 38, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation = 'linear')) # Continuous Activation for Regression Problems
model.add(Dense(1)) # Output

model.summary()

 
# load model


model.compile(optimizer='adam', loss='mean_squared_error')

epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 38, validation_split = 0.2)
# All information about the training
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
y_predict = model.predict(X_test)
plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel('True Values')
plt.ylabel('Model Predictions')
y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)
plt.plot(y_test_orig, y_predict_orig, "^", color = 'r')
plt.xlabel('True Values')
plt.ylabel('Model Predictions')
k = X_test.shape[1]
n = len(X_test)
n

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
score = model.evaluate(X,y, verbose=0)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
print("r2",r2)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
import pickle as pkl  
pickle_out1 = open("model.pkl", "wb")  
pkl.dump('p_model', pickle_out1)  
pickle_out1.close()  
print('Model Saved!')

#load model
