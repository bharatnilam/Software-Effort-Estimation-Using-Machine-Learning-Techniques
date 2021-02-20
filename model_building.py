# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:04:35 2021

@author: bharatnilam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ds = pd.read_csv('desharnais_clean.csv')

#choose relevant columns
ds.columns

ds_model = ds[['Length', 'Transactions', 'Entities', 'PointsAdjust', 'PointsNonAjust','Effort']]

#train test split
from sklearn.model_selection import train_test_split
X = ds_model.drop('Effort', axis=1)
y = ds_model['Effort']
   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=5)

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()

np.mean(cross_val_score(lm, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#support vector
from sklearn.svm import SVR

svr = SVR()

np.mean(cross_val_score(svr, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#decision tree classifier
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

np.mean(cross_val_score(dt, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#lasso regression
from sklearn import linear_model
lr = linear_model.Lasso()

np.mean(cross_val_score(lr, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#mlp
from sklearn.neural_network import MLPRegressor

iter = []
error = []
for i in range(1500,1800,50):
    iter.append(i)
    mlp = MLPRegressor(random_state=1, max_iter=i)
    error.append(np.mean(cross_val_score(mlp, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')))
plt.plot(iter,error)

np.mean(cross_val_score(MLPRegressor(random_state=1, max_iter=1550), X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#rf
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#kneighbors
from sklearn.neighbors import KNeighborsRegressor

kn = KNeighborsRegressor()

np.mean(cross_val_score(kn, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#tune models gridsearchcv
from sklearn.model_selection import GridSearchCV

#linear regressor tuning
lm_params = {'normalize':('True','False')}
gs_lm = GridSearchCV(lm, lm_params, scoring='neg_mean_absolute_error', cv=3, verbose=2)
gs_lm.fit(X_train, y_train)
gs_lm.best_params_
gs_lm.best_score_
gs_lm.best_estimator_

#svr tuning
svr_params = {'kernel':('linear', 'rbf'), 'gamma':('scale','auto'), 'C':range(1,10)}
gs_svr = GridSearchCV(svr, svr_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_svr.fit(X_train, y_train)
gs_svr.best_params_
gs_svr.best_score_
gs_svr.best_estimator_

#decision tree tuning
dt_params = {'criterion':('mse','mae'), 'splitter':('best','random'), 'min_samples_split':range(2,10), 'min_samples_leaf':range(1,10), 'max_features':('auto','sqrt'), 'random_state':range(30,42)}
gs_dt = GridSearchCV(dt, dt_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_dt.fit(X_train, y_train)
gs_dt.best_params_
gs_dt.best_score_
gs_dt.best_estimator_

#lasso tuning
lr_params = {'normalize':('True','False'), 'max_iter':range(1,10), 'random_state':range(0,42), 'selection':('cyclic','random')}
gs_lr = GridSearchCV(lr, lr_params, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=2)
gs_lr.fit(X_train, y_train)
gs_lr.best_params_
gs_lr.best_score_
gs_lr.best_estimator_

#test ensembles