# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 14:11:17 2021

@author: bharatnilam
"""

import pandas as pd

ch = pd.read_csv('china.csv')

ch.info()
ch.shape
ch.columns
ch.nunique(axis=0)
ch.describe()
ch.isnull().sum()

ch = ch.drop(columns = ['Dev.Type', 'ID'], axis = 1)

ch.to_csv('china_clean.csv', index = False)

