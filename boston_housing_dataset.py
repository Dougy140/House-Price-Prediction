# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 21:06:46 2021

@author: DOUGLAS
"""

#crim - per capita crime rate by town
#zn - proportion of residential land zoned for lots over 25,000 sq.ft 
#indus - proportion of non-retail business acres per town
#chas - charles river dummy variable(1 if tract bounds river ; 0 otherwise)
#nox - nitric oxides concentration(parts per 10 million)
#rm - average number of rooms per dwelling 
#age - proportion of owner occupied units built prior to 1940
#dis - weighted distances to five Boston employment centres
#rad - index of accessibility to radial accessibility to radial highways
#tax - full-value property-tax rate per $10,000
#ptratio- pupil-teacher ratio by town
#B-1000(Bk-0.63)^2 where Bk is the proportion of blacks by town
#lstat - %lower status of the population
#medv- media value of owner occupied homes in $1000's


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRARIO','B','LSTAT','MEDV']
data_path = "C:/Users/DOUGLAS/Desktop/jupyter/Boston/housing.csv"
df = pd.read_csv(data_path,header = None,delimiter= r'\s+',names=column_names)
#print(df.head())

#plotting the dataset
fig, axs = plt.subplots(ncols=7,nrows=2,figsize=(30,20))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.boxplot(y=k,data=df,ax=axs[index])
    index += 1

#plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=5.0)

#print(df.columns)

#finding the outliers
for k,v in df.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    iqr = q3 - q1
    v_col = v[(v<= q1 - 1.5 * iqr)| (v >= q3 + 1.5* iqr)]
    perc = np.shape(v_col)[0] * 100 / np.shape(df)[0]
    #print("column %s outlier s = %.2f%%" %(k,perc))

#remove MEDV outliers
df = df[~(df['MEDV'] >= 50)]
print(df.shape)

#visualization of medv and other features
fig, axs = plt.subplots(ncols=7,nrows=2,figsize=(30,20))
index = 0
axs = axs.flatten()
for k,v in df.items():
    sns.distplot(v,ax=axs[index])
    index += 1
#plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=5.0)

#plotting pairwise correlation on data
plt.figure(figsize=(30,20))
#sns.heatmap(df.corr().abs(),annot=True)


#plottinng the columns before plotting them against medv
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT','INDUS','NOX','PTRARIO','RM','TAX','DIS','AGE']
x = df.loc[:,column_sels]
y = df['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x),columns=column_sels)
fig,axs = plt.subplots(ncols=4,nrows=2,figsize=(20,10))
index = 0 
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y,x=x[k],ax=axs[i])
plt.tight_layout(pad=0.4,w_pad=0.5,h_pad=5.0)


#removing skewness of the data through log transformation
y = np.log1p(y)
for col in x.columns:
    if np.abs(x[col].skew() > 0.3):
        x[col] = np.log1p(x[col])

#lets try linear, ridge regression on dataset first
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np

l_regression = linear_model.LinearRegression()
kf = KFold(n_splits=10)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
scores = cross_val_score(l_regression,x_scaled,y,cv=kf,scoring='neg_mean_squared_error')
print("MSE: %0.2f (+/- %0.2f)" %(scores.mean(),scores.std()))

scores_map = {}
scores_map['LinearRegression'] = scores
l_ridge = linear_model.Ridge()
scores = cross_val_score(l_ridge, x_scaled, y, cv=kf,scoring='neg_mean_squared_error')
scores_map['Ridge'] = scores
print('MSE: %0.2f (+/- %0.2f)' %(scores.mean(),scores.std()))

##polynomial regression with l2 with degree for the best fit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
#for degree in range(2,6):
#    model = make_pipeline(PolynomialFeatures(degree=degree),linear_model.Ridge())
#    scores = cross_val_score(model, x_scaled, y, cv = kf, scoring = 'neg_mean_squared_error')
#    print('MSE: %0.2f (+/- %0.2f)' %(scores.mean(),scores.std()))

model = make_pipeline(PolynomialFeatures(degree=3),linear_model.Ridge())
scores = cross_val_score(model, x_scaled, y , cv=kf, scoring='neg_mean_squared_error')
scores_map['PolyRidge'] = scores
print('Polynomial')
print('MSE: %0.2f (+/- %0.2f)' %(scores.mean(),scores.std()))


#svr
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
svr_rbf = SVR(kernel='rbf', C = 1e3, gamma = 0.1)
scores = cross_val_score(svr_rbf, x_scaled, y , cv=kf, scoring='neg_mean_squared_error')
scores_map['SVR'] = scores
print('SVR')
print('MSE: %0.2f (+/- %0.2f)' %(scores.mean(),scores.std()))

#decision tree regressor
from sklearn.tree import DecisionTreeRegressor
desc_tr = DecisionTreeRegressor(max_depth=5)
scores = cross_val_score(desc_tr, x_scaled, y , cv=kf, scoring='neg_mean_squared_error')
scores_map['DecisionTReeRegressor'] = scores
print('DecisionTreeRegressor')
print('MSE: %0.2f (+/- %0.2f)' %(scores.mean(),scores.std()))

#knn regressor
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=7)
scores = cross_val_score(knn, x_scaled, y , cv=kf, scoring='neg_mean_squared_error')
scores_map['KNeighborsRegressor'] = scores
print('knn')
print('MSE: %0.2f (+/- %0.2f)' %(scores.mean(),scores.std()))


#gradient boosting
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(alpha = 0.9, learning_rate=0.05,max_depth=2,min_samples_leaf=5,min_samples_split=2,n_estimators=100,random_state=30)
scores = cross_val_score(gbr, x_scaled, y , cv=kf, scoring='neg_mean_squared_error')
scores_map['GradientBoostingRegressor'] = scores
print('GRadient boosting')
print('MSE: %0.2f (+/- %0.2f)' %(scores.mean(),scores.std()))


#lets plot k-fold results to see which model has better distribution
#plt.figure(figSize=(20,10))
scores_map = pd.DataFrame(scores_map)
sns.boxplot(data=scores_map)














