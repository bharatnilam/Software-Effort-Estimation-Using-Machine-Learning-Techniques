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
y = ds_model.Effort.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#linear regression
import statsmodels.api as sm
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, y_train)

#tune models gridsearchcv
#test ensembles