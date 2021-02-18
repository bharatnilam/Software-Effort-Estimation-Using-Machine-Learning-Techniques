# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:04:35 2021

@author: bharatnilam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ds = pd.read_csv('desharnais_clean.csv')

#choose relevasnt columns
ds.columns

ds_model = ds[['Length', 'Transactions', 'Entities', 'PointsAdjust', 'PointsNonAjust','Effort']]

#train test split
from sklearn.model_selection import train_test_split
X = ds_model.drop('Effort', axis=1)
y = ds_model['Effort']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

#linear regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

print(lm.score(X_test, y_test))

#support vector
from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV

svr_params = {'kernel' : ('linear', 'rbf'), 'gamma' : ('scale', 'auto')}

svr = SVR()

gs_svr = GridSearchCV(svr, svr_params, cv=3)
gs_svr.fit(X_train, y_train)
print(gs_svr.score(X_test, y_test))

gs_svr.best_params_

np.mean(cross_val_score(gs_svr, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#tune models gridsearchcv
#test ensembles