# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 02:32:17 2022

@author: GHLADIN SHEBAC S R
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('C:/Users/GHLADIN SHEBAC S R/OneDrive/Desktop/Deploy Rice/trained_model.sav', 'rb'))


#creating a function for Prediction

def rice_prediction(input_data):

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 1):
      return 'The Rice is Gonen'

    else:
      return 'The Rice is Jasmine'
  
    
  
    
def main():
    
    
    #giving a title
    st.title('Rice Prediction Web App')
    
    #getting the input data from the user
    
    Area = st.text_input('Area of Rice')
    MajorAxisLength = st.text_input('MajorAxisLength of Rice')
    MinorAxisLength = st.text_input('MinorAxisLength of Rice')
    Eccentricity = st.text_input('Eccentricity of Rice')
    ConvexArea = st.text_input('ConvexArea of Rice')
    EquivDiameter = st.text_input('EquivDiameter of Rice')
    Extent = st.text_input('Extent of Rice')
    Perimeter = st.text_input('Perimeter of Rice')
    Roundness = st.text_input('Roundness of Rice')
    AspectRation = st.text_input('AspectRation of Rice')

    
    #code for prediction
    diagnosis = ''
    
    #creating a button fro Prediction
    
    if st.button('Rice Test Result'):
        diagnosis = rice_prediction(np.array([Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation],dtype=np.float32))
        
        
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
        