# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:03:33 2021

@author: bharatnilam
"""

import pandas as pd
import numpy as np

ch = pd.read_csv('china_clean.csv')

#choose relevant columns
ch_feat = ch[['Added', 'File', 'Input', 'NPDR_AFP', 'NPDU_UFP', 'Output', 'PDR_UFP', 'Effort']]

#train test split
from sklearn.model_selection import train_test_split
X = ch_feat.drop('Effort', axis = 1)
y = ch_feat['Effort']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#lasso regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

lr = Lasso()

np.mean(cross_val_score(lr, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#elastic net
from sklearn.linear_model import ElasticNet

en = ElasticNet()

np.mean(cross_val_score(en, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#ridge regression
from sklearn.linear_model import Ridge

rr = Ridge()

np.mean(cross_val_score(rr, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#svr linear
from sklearn.svm import SVR

svrl = SVR(kernel='linear')

np.mean(cross_val_score(svrl, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

#svr rbf
from sklearn.svm import SVR

svrr = SVR(kernel='rbf')

np.mean(cross_val_score(svrr, X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))

