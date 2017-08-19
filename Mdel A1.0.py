import pandas as pd

import numpy as np

import datalab.bigquery as bq

import tensorflow as tf

import io

import requests

##

Temptable = pd.DataFrame(columns=['DateATL','AvgTATL','AvgTORD','AvgTDFW','AvgTJFK'])

for j in range (2010,2017):
  url="https://www.wunderground.com/history/airport/ATL/{}/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(j)
  s=requests.get(url).content
  ATL1=pd.read_csv(io.StringIO(s.decode('utf-8')))
  ATL = ATL1[["EST", "Mean TemperatureF"]]

  url="https://www.wunderground.com/history/airport/ORD/2010/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(j)
  s=requests.get(url).content
  ORD1=pd.read_csv(io.StringIO(s.decode('utf-8')))
  ORD = ORD1[["CST", "Mean TemperatureF"]]

  url="https://www.wunderground.com/history/airport/DFW/2010/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(j)
  s=requests.get(url).content
  DFW1=pd.read_csv(io.StringIO(s.decode('utf-8')))
  DFW = DFW1[["CST", "Mean TemperatureF"]]

  url="https://www.wunderground.com/history/airport/JFK/2010/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(j)
  s=requests.get(url).content
  JFK1=pd.read_csv(io.StringIO(s.decode('utf-8')))
  JFK = JFK1[["EST", "Mean TemperatureF"]]

  for i in range(0, 10):
    DateATL = ATL ['EST'].ix[i]
    #TempATL = ATL ['Mean TemperatureF'].ix[i]
    a = abs(72-np.mean([ATL['Mean TemperatureF'].ix[(i-1):(i+1)]]))
    #DateORD = ORD ['CST'].ix[i]
    #TempORD = ORD ['Mean TemperatureF'].ix[i]
    b = abs(72-np.mean([ORD['Mean TemperatureF'].ix[(i-1):(i+1)]]))
    #DateDFW = DFW ['CST'].ix[i]
    #TempDFW = DFW ['Mean TemperatureF'].ix[i]
    c = abs(72-np.mean([DFW['Mean TemperatureF'].ix[(i-1):(i+1)]]))
    #DateJFK = JFK ['EST'].ix[i]
    #TempJFK = JFK ['Mean TemperatureF'].ix[i]
    d = abs(72-np.mean([JFK['Mean TemperatureF'].ix[(i-1):(i+1)]]))
    Temptable = Temptable.append({'DateATL':DateATL,'AvgTATL':a,'AvgTORD':b,'AvgTDFW':c,'AvgTJFK':d},ignore_index=True)

print Temptable

##

Temptable = pd.DataFrame(columns=['DateATL','ScoreATL','AvgTATL','DateORD','ScoreORD','AvgTORD','DateDFW','ScoreDFW','AvgTDFW','DateJFK','ScoreJFK','AvgTJFK'])

for i in range(0, 10):
  DateATL = ATL ['EST'].ix[i]
  TempATL = ATL ['Mean TemperatureF'].ix[i]
  a = abs(72-np.mean([ATL['Mean TemperatureF'].ix[(i-1):(i+1)]]))
  DateORD = ORD ['CST'].ix[i]
  TempORD = ORD ['Mean TemperatureF'].ix[i]
  b = abs(72-np.mean([ORD['Mean TemperatureF'].ix[(i-1):(i+1)]]))
  DateDFW = DFW ['CST'].ix[i]
  TempDFW = DFW ['Mean TemperatureF'].ix[i]
  c = abs(72-np.mean([DFW['Mean TemperatureF'].ix[(i-1):(i+1)]]))
  DateJFK = JFK ['EST'].ix[i]
  TempJFK = JFK ['Mean TemperatureF'].ix[i]
  d = abs(72-np.mean([JFK['Mean TemperatureF'].ix[(i-1):(i+1)]]))
  Temptable = Temptable.append({'DateATL':DateATL,'ScoreATL':TempATL,'AvgTATL':a,'DateORD':DateORD,'ScoreORD':TempORD,'AvgTORD':b,
                                'DateDFW':DateDFW,'ScoreDFW':TempDFW,'AvgTDFW':c,'DateJFK':DateJFK,'ScoreJFK':TempJFK,'AvgTJFK':d},ignore_index=True)

#for i,d in enumerate(data_you_want):
    #if (i % 600) == 0:
        #avg_for_day = np.mean(data_you_want[i - 600:i])
        #daily_averages.append(avg_for_day)  
  
#print a
print Temptable
# goes through date 2-1-2011

##

url="https://www.wunderground.com/history/airport/ATL/2010/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779"
s=requests.get(url).content
ATL1=pd.read_csv(io.StringIO(s.decode('utf-8')))
ATL = ATL1[["EST", "Mean TemperatureF"]]

url="https://www.wunderground.com/history/airport/ORD/2010/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779"
s=requests.get(url).content
ORD1=pd.read_csv(io.StringIO(s.decode('utf-8')))
ORD = ORD1[["CST", "Mean TemperatureF"]]

url="https://www.wunderground.com/history/airport/DFW/2010/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779"
s=requests.get(url).content
DFW1=pd.read_csv(io.StringIO(s.decode('utf-8')))
DFW = DFW1[["CST", "Mean TemperatureF"]]

url="https://www.wunderground.com/history/airport/JFK/2010/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779"
s=requests.get(url).content
JFK1=pd.read_csv(io.StringIO(s.decode('utf-8')))
JFK = JFK1[["EST", "Mean TemperatureF"]]

#print JFK

##

  Temptable = Temptable.append({
        'return_positive':return_positive,
        'return_negative':return_negative,
        'HighbyClose':HighbyClose,
        'LowbyClose':LowbyClose,
        'VolumebyAvgVolume':VolumebyAvgVolume,
        '1DayChange':OneDayChange,
        '10DayChange':TenDayChange,
        'CARZrat':CARZrat,
        'NILSYrat':NILSYrat,
        'Settle':Settle
        },ignore_index=True)

##

for i in range(2010,2012):
  
  url="https://www.wunderground.com/history/airport/ATL/{}/1/1/CustomHistory.html?dayend=1&monthend=12&yearend=2016&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(i)
  s=requests.get(url).content
  ATL1=pd.read_csv(io.StringIO(s.decode('utf-8')))
  ATL = ATL1[["EST", "Mean TemperatureF"]]
  
print ATL

##

merged_inner = pd.merge(left=DataPALLonly,right=DataCARZ, left_on='Date', right_on='Date')
DataPALL1 = pd.merge(left=merged_inner,right=DataNILSY, left_on='Date', right_on='Date')
DataPALL = pd.merge(left=DataPALL1,right=Futures, left_on='Date', right_on='Date')
DataPALL.columns = ["Date", "Open_x", "High_x", "Low_x","Close_x","Volume_x","Adj Close_x","8","9","10","11","12","Adj Close_y","14","15","16","17","18","Adj Close","20","21","22","23","24","Settle","26","27"]

#print DataPALL

##

Temptable = pd.DataFrame(columns=['Date','Score'])

for i in range(0, 20):
  Date = ATL ['EST'].ix[i]
  Score = abs(72-ATL['Mean TemperatureF'].ix[i+10])
  Temptable = Temptable.append({'Date':Date,'Score':Score},ignore_index=True)

