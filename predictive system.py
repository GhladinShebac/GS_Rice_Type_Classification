# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 02:01:29 2022

@author: GHLADIN SHEBAC S R
"""

import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('C:/Users/GHLADIN SHEBAC S R/OneDrive/Desktop/Deploy Rice/trained_model.sav', 'rb'))

input_data = (6134,153.0819809,51.59060559,0.9415000217,6283,88.37449501,0.4899752376,338.613,0.6722741011,2.967245279)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Rice is Gonen')

else:
  print('The Rice is Jasmine')

