# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 19:19:06 2022

@author: kaush
"""

import pickle
import numpy as np
import streamlit as st
import statsmodels.api as sm


pickle_in = open("classifier.pkl","rb")
data = pickle.load(pickle_in)



def sales_prediction(input_data):
    
    model = sm.tsa.statespace.SARIMAX(data, order=(0,1,1), seasonal_order=(0,0,0,12), 
                                 enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit()
   
    pred = result.get_prediction(start=input_data, dynamic=False)
    pred_ci = pred.conf_int()
    
   
    with st.container():
        st.subheader('Predicted sales from the Input Date')
        ax=data[input_data:].plot(label='observed')
        pred.predicted_mean.plot(ax=ax, label='forecast', alpha=0.7, figsize=(15,5))
        ax.fill_between(pred_ci.index,
                        pred_ci.iloc[:, 0],
                        pred_ci.iloc[:, 1], color='k', alpha=.2)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
        y_forecasted = pred.predicted_mean
        y_truth = data[input_data:]
    
        rmse = ((y_forecasted - y_truth)**2).mean()
        rmse_show =  np.sqrt(rmse)
        st.metric('The Mean Squared Error is:', rmse_show)
        
    
 
    
def main():
    
    st.title("Dataset-2 ARIMA Model")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h4 style="color:white;text-align:center;">In this model we will take input parameter as date in the format YYYY-MM-DD and give output as a graph of future predictions from the given Input Date by the user along with the RMSE value of the ARIMA model </h4>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.write("")
    input_data = st.text_input("Enter a Date in the format YYYY-MM-DD") 
    button1 = st.button("Predict")
        
    if button1:
        sales_prediction(input_data)
        
      
if __name__ == '__main__':
    main()
