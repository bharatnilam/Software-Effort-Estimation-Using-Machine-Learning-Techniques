# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 22:03:07 2021

@author: bharatnilam
"""

import requests
from data_input import data_in

URL = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}
data = {'input': data_in}

r = requests.get(url = URL, headers = headers, json = data)

r.json()