# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 16:00:57 2021

@author: bharatnilam
"""

import pandas as pd

ds = pd.read_csv('desharnais.csv')

#mean calculation
mean_TeamExp = ds['TeamExp'].mean()
mean_ManagerExp = ds['ManagerExp'].mean()

#fill missing values
ds['TeamExp'].fillna(int(mean_TeamExp), inplace = True)
ds['ManagerExp'].fillna(int(mean_ManagerExp), inplace = True)

#language field
lang = ds['Language'].apply(lambda x: int(x.replace("b'","").replace("'","")))
ds['Language'] = lang

ds.to_csv('desharnais_clean.csv', index = False)
