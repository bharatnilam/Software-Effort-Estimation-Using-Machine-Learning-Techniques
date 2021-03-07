# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 01:42:32 2021

@author: bharatnilam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score

mw = pd.read_csv('maxwell.csv')

#choose features
mw_model = mw[['App', 'Har', 'Nlan', 'T07', 'T11', 'T12', 'Duration', 'Size', 'Effort']]

#train test split
X = mw_model.drop('Effort', axis = 1)
y = mw_model['Effort']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#linear regression
from sklearn.linear_model import LinearRegression
lr= LinearRegression()

np.mean(cross_val_score(lr, X_train, y_train, cv = 3, scoring = 'neg_mean_absolute_error'))


#tune models
from sklearn.model_selection import GridSearchCV
lr_params = {'normalize':('True','False')}
gs_lr = GridSearchCV(lr, lr_params, scoring='neg_mean_absolute_error', cv=3, verbose=2)
gs_lr.fit(X_train, y_train)
gs_lr.best_params_
gs_lr.best_score_
gs_lr.best_estimator_
#test ensembles