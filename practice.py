# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:00:47 2022

@author: Tejas
"""

import pandas as pd
from pandas import read_csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import save_model
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
def get_data_from_excel():
    df=pd.read_excel(
       io='Book.xlsx',
       engine='openpyxl',
       sheet_name='Sheet1',
       usecols='A:R',
       nrows=20,
    )
    return df
df =get_data_from_excel()

#sns.pairplot(df)
#print(df.columns)
#del df ['Date']
#print(df.count(0))
#print(df.shape)

#Define the independent and dependent variables
df
y= df[['Total sales']].values#dependent variable is Decision
x= df[['Total Production']].values
x=x.reshape(-1,1)
y=y.reshape(-1,)
#print(x.shape)
#print(y.shape)

model1 = LinearRegression().fit(x, y)
#print(x)
#print(y)
print(model1.intercept_)
print(model1.coef_)

y_pred1 = model1.predict(x)
print(y_pred1.shape)

r_sq=model1.score(x, y)
print(r_sq)

#print("predicted response:",y_pred)



def get_data_from_excel():
    df2=pd.read_excel(
       io='Book.xlsx',
       engine='openpyxl',
       sheet_name='Sheet3',
       usecols='A:R',
       nrows=4,
    )
    return df2
df2 =get_data_from_excel()

x_new=df2['Total Production'].values
x_new=x_new.reshape(-1,1)
print(x_new.shape)
print(x_new)
y_new = model1.predict(x_new)

print(y_new)

y_new=df2['Total Production'].values
y_pred=model1.predict(x_new)
print("these are prediction")
#csv = pd.DataFrame(y_pred, columns=["Predicted 2021 values"])
#csv.to_csv("pred.csv", index=False)


print(x_new.shape)
print(y_new.shape)
print(y_pred.shape)

plt.scatter(y_new, y_pred)
plt.show()
sns.regplot(x=y_new,
            y=y_pred,
            data=df2)
plt.show()
#print(y_pred1)
k = x_new.shape[1] # Number of Features
n = len(x_new) # Number of Cases
print("Features:",k)
print("Cases:",n)
y_new_new=df2['Total sales'].values
x_train,x_test,y_train,y_test = train_test_split(x_new,y_new_new,test_size=0.2)
# Shape Checking
print(x_train.shape)
# Shape Checking
print(x_test.shape)
#"part 5 sk learn"

# Using Linear Regression Model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)
# Get the predictions
y_predict = regressor.predict(x_test)
y_predict.shape

# Get the Values "before" scaling

# Number of Features and Cases
k = x_test.shape[1] # Number of Features
n = len(x_new) # Number of Cases
print("Features:",k)
print("Cases:",n)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print(y_test.shape)
print(y_predict.shape)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE=mean_absolute_error(y_test, y_predict)
r2=r2_score(y_test, y_predict)
adj_r2 = 1 - (1 - r2) * (n -1) / (n - k -1)
# Evaluation Results Printing
print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

#print("which column")
#print(df(model.intercept_+model.coef_*x_test))


#plot

plt.show()
corr = df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True)
plt.show()


y = y.reshape(-1,1)
y.shape
print(x,y)
del df['Date']

# Columns Check
df.columns
# Check the Weights for the various Features
#list(zip(['Total Production'], regressor.coef_[0])) 

#main features

#Scaling the data numerical data before feeding the model
#from sklearn.preprocessing import MinMaxScaler




import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Ftrl
model = keras.Sequential()
model.add(Dense(32, input_dim = 1, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation = 'linear')) # Continuous Activation for Regression Problems
model.add(Dense(1)) # Output

model.summary()

 
# load model

y2= df2[['Total sales']].values#dependent variable is Decision
x2= df2[['Total Production']].values
x2=x2.reshape(-1,1)
print(x2.shape)
model.compile(optimizer='Ftrl', loss='mean_squared_error')
print(y_train.shape)
epochs_hist = model.fit(x, y, epochs = 225, batch_size = 1, validation_split = 0.4)
# All information about the training
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])
y_predict = model.predict(x)
print(y_predict)
plt.plot(y_new, "^", color = 'g')
plt.show()


k = x_train.shape[1]
n = len(x_train)
n

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
score = model.evaluate(x,y, verbose=1)

RMSE = float(format(np.sqrt(mean_squared_error(y, y_predict)),'.3f'))
MSE = mean_squared_error(y, y_predict)
MAE = mean_absolute_error(y, y_predict)
r2 = r2_score(y, y_predict)
print("r2",r2)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 
print(x.shape)
print(y.shape)
print(y_predict.shape)
plt.scatter(y, y_predict)
plt.show()
sns.regplot(x=y,
            y=y_predict,
            data=df2)
plt.show()

#load model