# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 14:54:50 2021

@author: bharatnilam
"""

from scipy.io import arff
import pandas as pd

data1 = arff.loadarff('Maxwell/maxwell.arff')
mw = pd.DataFrame(data1[0])

data2 = arff.loadarff('Desharnais/desharnais.arff')
ds = pd.DataFrame(data2[0])

data3 = arff.loadarff('China/china.arff')
ch = pd.DataFrame(data3[0])

mw.to_csv('Maxwell/maxwell.csv', index = False)

ds.to_csv('Desharnais/desharnais.csv', index = False)

ch.to_csv('China/china.csv', index = False)
