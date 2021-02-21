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
from sklearn.svm import LinearSVR

svr = LinearSVR()

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

mlp = MLPRegressor(max_iter=3220)

np.mean(cross_val_score(MLPRegressor(), X_train, y_train, cv=3, scoring='neg_mean_absolute_error'))
    
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
svr_params = {'loss':('epsilon_insensitive','squared_epsilon_insensitive'), 'random_state':range(0,42), 'max_iter':range(4000,6000,100)}
gs_svr = GridSearchCV(svr, svr_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_svr.fit(X_train, y_train)
gs_svr.best_params_
gs_svr.best_score_
gs_svr.best_estimator_

#decision tree tuning
dt_params = {'criterion':('mse','mae'), 'splitter':('best','random'), 'max_features':('auto','sqrt','log2'), 'random_state':range(0,42), 'min_samples_split':range(5,10)}
gs_dt = GridSearchCV(dt, dt_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_dt.fit(X_train, y_train)
gs_dt.best_params_
gs_dt.best_score_
gs_dt.best_estimator_

#lasso tuning
lr_params = {'normalize':('True','False'), 'random_state':range(0,42), 'selection':('cyclic','random')}
gs_lr = GridSearchCV(lr, lr_params, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1, verbose=2)
gs_lr.fit(X_train, y_train)
gs_lr.best_params_
gs_lr.best_score_
gs_lr.best_estimator_

#mlp tuning
mlp_params = {'solver':('lbfgs','adam'), 'activation':['relu'], 'random_state':range(0,42,2), 'max_iter':range(2500,3500,100)}
gs_mlp = GridSearchCV(mlp, mlp_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_mlp.fit(X_train, y_train)
gs_mlp.best_params_
gs_mlp.best_score_
gs_mlp.best_estimator_

#rf tuning
rf_params = {'n_estimators':range(15,25), 'criterion':('mae','mse'), 'max_features':('auto','sqrt','log2'), 'random_state':range(0,42,2)}
gs_rf = GridSearchCV(rf, rf_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_rf.fit(X_train, y_train)
gs_rf.best_params_
gs_rf.best_score_
gs_rf.best_estimator_

#kn tuning
kn_params = {'n_neighbors':range(1,36), 'weights':('uniform','distance'), 'algorithm':('auto','ball_tree','kd_tree','brute')}
gs_kn = GridSearchCV(kn, kn_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_kn.fit(X_train, y_train)
gs_kn.best_params_
gs_kn.best_score_
gs_kn.best_estimator_

#test ensembles
tpred_lm = gs_lm.best_estimator_.predict(X_test)
tpred_svr = gs_svr.best_estimator_.predict(X_test)
tpred_dt = gs_dt.best_estimator_.predict(X_test)
tpred_lr = gs_lr.best_estimator_.predict(X_test)
tpred_mlp = gs_mlp.best_estimator_.predict(X_test)
tpred_rf = gs_rf.best_estimator_.predict(X_test)
tpred_kn = gs_kn.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, tpred_lm)
mean_absolute_error(y_test, tpred_svr)
mean_absolute_error(y_test, tpred_dt)
mean_absolute_error(y_test, tpred_lr)
mean_absolute_error(y_test, tpred_mlp)
mean_absolute_error(y_test, tpred_rf)
mean_absolute_error(y_test, tpred_kn)

#bagging regressor
from sklearn.ensemble import BaggingRegressor
br = BaggingRegressor(base_estimator=gs_rf.best_estimator_)
br_params = {'n_estimators':range(15,25), 'random_state':range(0,42,2)}
gs_br = GridSearchCV(br, br_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_br.fit(X_train, y_train)
gs_br.best_params_
gs_br.best_score_
gs_br.best_estimator_


#adaboost regressor
from sklearn.ensemble import AdaBoostRegressor
abr = AdaBoostRegressor(base_estimator=gs_mlp.best_estimator_)
abr_params = {'n_estimators':[52], 'random_state':range(32,42)}
gs_abr = GridSearchCV(abr, abr_params, scoring='neg_mean_absolute_error', cv=3, verbose=2, n_jobs=-1)
gs_abr.fit(X_train, y_train)
gs_abr.best_params_
gs_abr.best_score_
gs_abr.best_estimator_

tpred_br = gs_br.best_estimator_.predict(X_test)
tpred_abr = gs_abr.best_estimator_.predict(X_test)

mean_absolute_error(y_test, tpred_br)
mean_absolute_error(y_test, tpred_abr)

#pickle
import pickle
pickl = {'model': gs_abr.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']
model.predict(X_test.iloc[1,:].values.reshape(1,-1))

list(X_test.iloc[1,:].values)