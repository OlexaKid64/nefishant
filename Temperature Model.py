import pandas as pd

import numpy as np

import datalab.bigquery as bq

import tensorflow as tf

import io

import requests

import json

import urllib2

import datetime as dt

##

Temptable = pd.DataFrame(columns=['date','todayT'])

dt.datetime.today().strftime("%m/%d/%Y")
day = np.int(dt.datetime.today().strftime("%d"))
month = np.int(dt.datetime.today().strftime("%m"))
numdays = np.int(dt.datetime.today().strftime("%j"))-1


url="https://www.wunderground.com/history/airport/ATL/2017/1/1/CustomHistory.html?dayend={}&monthend={}&yearend=2017&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(day,month)
s=requests.get(url).content
ATL1=pd.read_csv(io.StringIO(s.decode('utf-8')))
ATL = ATL1[["EDT", "Mean TemperatureF"]]

f = urllib2.urlopen('http://api.wunderground.com/api/e4026bac17a9a878/forecast10day/q/GA/ATL.json')
json_string = f.read()
parsed_json = json.loads(json_string)

for i in range(0, numdays):
    date = ATL ['EDT'].ix[i]
    todayT = ATL ['Mean TemperatureF'].ix[i]
#    a = np.mean([ATL['Mean TemperatureF'].ix[(i):(i+10)]])
    Temptable = Temptable.append({'date':date,'todayT':todayT
                                  #,'AvgTATL':a
                                 },ignore_index=True)  

print Temptable

##

f = urllib2.urlopen('http://api.wunderground.com/api/e4026bac17a9a878/forecast10day/q/GA/ATL.json')
json_string = f.read()
parsed_json = json.loads(json_string)
for i in range(0,10):
    year = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['year']
    month = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['month']
    day = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['day']
    date = dt.date(year,month,day)
    ForecastH = np.float32(parsed_json['forecast']['simpleforecast']['forecastday'][i]['high']['fahrenheit'])
    ForecastL = np.float32(parsed_json['forecast']['simpleforecast']['forecastday'][i]['low']['fahrenheit'])
    Mean =(ForecastH+ForecastL)/2
 
    Temptable = Temptable.append({'date':date,'todayT':Mean
                                  #,'AvgTATL':a
                                 },ignore_index=True) 


#df to_csv('file.csv')
#Temptable['todayT'].plot()
#plot.show()
#print Temptable

##

ATLtemptbl = pd.DataFrame(columns=['date','todayT','10daymean'])

for i in range(0, numdays):
    date = Temptable ['date'].ix[i]
    todayT = Temptable ['todayT'].ix[i]
    a = np.mean([Temptable['todayT'].ix[(i):(i+10)]])
    ATLtemptbl = ATLtemptbl.append({'date':date,'todayT':todayT,'10daymean':a},ignore_index=True)  

print ATLtemptbl
ATLtemptbl.plot()

#df to_csv('file.csv')
#df['Adj Close'].plot()
#plot.show()

##

