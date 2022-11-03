This project aims at forecasting closing prices of Criptocurrencies, among other time series. Mainly, its intent is to find out if a currency price will either drop or increase in the following days (as simple buy/sell flags). 

The timeseries.py serves as a module for forecasting automation. It contains functions for preparing the data, including pacf-based lag  


To do:

timeseries.py:
- include the fit_model and prepare_data LSTM-related functions from the ARIMA-ATTENTION notebook 
- include an automatic periodogram analysis
- Explore prophet and darts functionalities 

ARIMA-ATTENTION-LSTM.ipynb:
- Make forecasts using ARIMA residuals (based on https://github.com/zshicode/Attention-CLX-stock-prediction)
- Separate data in batches for stateful LSTM training
- Train the model with different securities (stateful LSTM)
