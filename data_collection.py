# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:54:50 2021

@author: bharatnilam
"""

from scipy.io import arff
import pandas as pd

data1 = arff.loadarff('maxwell.arff')
mw = pd.DataFrame(data1[0])

mw.to_csv('maxwell.csv')

data2 = arff.loadarff('desharnais.arff')
ds = pd.DataFrame(data2[0])

ds.to_csv('desharnais.csv')
