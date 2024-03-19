# Stock-Forcast
The stock data used is a pickle file containing a list of dataframes belonging to 2000 distinct stocks. Each dataframe contain trading records of 2202 consecutive days of that stock. A trading record contain Open, Low, High, Close and Volume instances of that day. Also, these records have already been column-wise normalized. 
The task is to predict whether the close price on a particular day increase, decrease or almost-unchanged compared to the close price of its previous day using the trading records of previous days. So, the labels are like ‘increase’, ‘decrease’ and ‘no big change’.
I mainly considered three approaches – 
1.	Using OLHV values of the past trading records to predict current day’s change in the Close price (increase or decrease or no much change?).
2.	Using just the Close price of past days to make a prediction about current day’s Close price.

### ** Able to acheive a precision of 50% on the test set with XGBoost and using just the close values **
Path to the file - usingCloseValues/usingXGBoost/XGBoostWith32Features-CloseValues.ipynb
