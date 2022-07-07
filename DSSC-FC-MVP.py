

# load library
import streamlit as st
import numpy as np
import pandas as pd
import pandas_profiling
from pandas_datareader import data as pdr
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import math
import statistics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime as dt
import yfinance as yf
import datetime
from data_statistics import DSSC_FC_MVP_Visualization as vz

# print title of web app
st.title("Financial Consultation Stock Market Analysis and Prediction")
st.markdown(">Welcome to the Website")

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

# Load data from yahoo finance.
start=dt.date(2021,1,1)
end=dt.date(2021,12,31)
data=pdr.get_data_yahoo("IBM", start, end)

data1=pdr.get_data_yahoo("MSFT", start, end)

#fill nan vale with next value within columns
data.fillna(method="ffill",inplace=True)

# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')


def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

csv = convert_df(data)

st.download_button(
     label="Download data as CSV",
     data=csv,
     file_name='data.csv',
     mime='text/csv',
 )


# create checkbox
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# show the description of data
st.subheader('Detail description about Datasets:-')
def describe(input_df):
  describe = input_df.describe()
  describe = describe.append(input_df.median().to_frame().T.rename(index={0: 'median'}))
  describe = describe.append(input_df.var().to_frame().T.rename(index={0: 'var'}))
  describe.loc['range'] = describe.loc['max']-describe.loc['min']
  return describe

descrb=describe(data)
st.write(descrb)


st.subheader('Interactive Dashboards to the user for better interaction and understanding the data results ')
pr=ProfileReport(data)
st_profile_report(pr)



#create new columns like year, month, day
data["Year"]=data.index.year
data["Month"]=data.index.month
data["Weekday"]=data.index.day_name()

# dislay graph of open and close column
st.subheader('Graph of Close & Open:-')
st.line_chart(data[["Open","Close"]])

# display plot of Adj Close column in datasets
st.subheader('Graph of Adjacent Close:-')
st.line_chart(data['Adj Close'])

# display plot of volume column in datasets
st.subheader('Graph of Volume:-')
st.line_chart(data['Volume'])

# candlestick ibm
st.subheader('IBM Stock:-')

candlestick = plt.figure(figsize = (15,7))

last20_ibm = data[:20]

up = last20_ibm[last20_ibm.Close >= last20_ibm.Open]
  
down = last20_ibm[last20_ibm.Close < last20_ibm.Open]
  
col1 = 'green'

col2 = 'red'
  
width = .3
width2 = .03
  
plt.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
plt.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
plt.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)
  
plt.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
plt.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
plt.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)

plt.xticks(rotation=30, ha='right')
plt.title('IBM Stock')  

st.pyplot(candlestick)

# MSFT Stock candlestick
st.subheader('MSFT Stock:-')
candlestick1 = plt.figure(figsize = (15,7))

last20_msft = data1[:20]

up = last20_msft[last20_msft.Close >= last20_msft.Open]
  
down = last20_msft[last20_msft.Close < last20_msft.Open]
plt.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
plt.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
plt.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)
  
plt.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
plt.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
plt.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)

plt.xticks(rotation=30, ha='right')
plt.title('MSFT Stock') 

st.pyplot(candlestick1)

st.subheader('IBM vs MSFT')
fig = plt.figure(figsize = (15,7))
plt.plot(data['Open'], label = 'ibm')
plt.plot(data1['Open'], label = 'msft')
plt.title('Stock Prices of IBM, MSFT')
plt.legend()
st.pyplot(fig)

# market cap
st.subheader('Marketcap IBM vs MSFT')
marketcap = plt.figure(figsize = (15,7))

data = vz.marketcap(data)
data1 = vz.marketcap(data1)
data['MarktCap'].plot(label = 'IBM', figsize = (15,7))
data1['MarktCap'].plot(label = 'MSFT')
plt.ylabel('Billions $')
plt.title('Market Cap')
plt.legend()

st.pyplot(marketcap)
# create new cloumn for data analysis.
data['HL_PCT'] = (data['High'] - data['Low']) / data['Close'] * 100.0
data['PCT_change'] = (data['Close'] - data['Open']) / data['Open'] * 100.0
data = data[['Adj Close', 'HL_PCT', 'PCT_change', 'Volume']]

# display the new dataset after modificaton
st.subheader('Newly format DataSet:-')
st.dataframe(data.tail(500))

forecast_col = 'Adj Close'
forecast_out = int(math.ceil(0.01 * len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)

X = np.array(data.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
data.dropna(inplace=True)
y = np.array(data['label'])

# split dataset into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

# display the accuracy of forecast value.
st.subheader('Accuracy:')
st.write(confidence)

forecast_set = clf.predict(X_lately)
data['Forecast'] = np.nan

last_date = data.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)]+[i]
    last_date = data.iloc[-1].name
    dti = pd.date_range(last_date, periods=forecast_out+1, freq='D')
    index = 1
for i in forecast_set:
    data.loc[dti[index]] = [np.nan for _ in range(len(data.columns)-1)] + [i]
    index +=1

# display the forecast value.
st.subheader('Forecast value :-')
st.dataframe(data.tail(50))

# display the graph of adj close and forecast columns
st.subheader('Graph of Adj Close and Forecast :-')
st.line_chart(data[["Adj Close","Forecast"]])

