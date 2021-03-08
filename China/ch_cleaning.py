# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:30:14 2021

@author: bharatnilam
"""

import pandas as pd
import numpy as np

ch = pd.read_csv('China/china.csv')

ch.head()

ch = ch.drop(['Dev.Type', 'ID'], axis = 1)

ch.to_csv('China/china_clean.csv', index = False)