# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:19:06 2022

@author: kaush
"""
pip install statsmodels

import pickle
import streamlit as st
import statsmodels.api as sm

pickle_in = open("classifier.pkl","rb")
data = pickle.load(pickle_in)

print("File running")

def sales_prediction(input_data):
    
    model = sm.tsa.statespace.SARIMAX(data, order=(0,1,1), seasonal_order=(0,0,0,12), 
                                 enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()
   
    pred = result.get_prediction(start=input_data, dynamic=False)
    pred_ci = pred.conf_int()
    
    #return st.success("Prediction:", pred_ci)
    
    ax=data[-10:].plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead forecast', alpha=0.7, figsize=(15,5))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    
    with st.container():
        st.write('This graph is predicting sales of 10 days from the Input Date')
        ax=data[-10:].plot(label='observed')
        pred.predicted_mean.plot(ax=ax, label='One-step ahead forecast', alpha=0.7, figsize=(15,5))
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    
    
 
    
def main():
    
    st.title("Dataset-2 ARIMA Model")
    
    input_data = st.text_input("Enter a Date in the format YYYY-MM-DD") 
    button1 = st.button("Predict")
        
    if button1:
        sales_prediction(input_data)
        
      
if __name__ == '__main__':
    main()
