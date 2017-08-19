import pandas as pd

import numpy as np

import datalab.bigquery as bq

import tensorflow as tf

import io

import requests

import json

import urllib.request

import datetime as dt

from datetime import datetime, timedelta

##

Temptable = pd.DataFrame(columns=['date','todayT'])

for j in range (2007,2017):
  url="https://www.wunderground.com/history/airport/ATL/{}/1/1/CustomHistory.html?dayend=1&monthend=3&yearend=2017&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(j)
  s=requests.get(url).content
  ATL1=pd.read_csv(io.StringIO(s.decode('utf-8')))
  ATL = ATL1[["EST", "Mean TemperatureF"]]


  for i in range(0, 365):
    DateATL = ATL ['EST'].ix[i]
    a = np.mean(ATL['Mean TemperatureF'].ix[i])

    Temptable = Temptable.append({'date':DateATL,'todayT':a},ignore_index=True)

#print Temptable
Temptable.plot()

##

dt.datetime.today().strftime("%m/%d/%Y")
day = np.int(dt.datetime.today().strftime("%d"))
month = np.int(dt.datetime.today().strftime("%m"))
numdays = np.int(dt.datetime.today().strftime("%j"))-1


url="https://www.wunderground.com/history/airport/ATL/2017/1/1/CustomHistory.html?dayend={}&monthend={}&yearend=2017&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(day,month)
s=requests.get(url).content
ATL1=pd.read_csv(io.StringIO(s.decode('utf-8')))
ATL = ATL1[["EDT", "Mean TemperatureF"]]

for i in range(0, numdays):
  date = ATL ['EDT'].ix[i]
  todayT = ATL ['Mean TemperatureF'].ix[i]
  Temptable = Temptable.append({'date':date,'todayT':todayT},ignore_index=True)  

print (Temptable)
Temptable.plot()

##
##Now we will add ten days of future forecast data to the file.
##We need future data because we will calculate the mean temperature
##10 days into the future as our model input.
##NEED TO FIX THIS CODED BECAUSE IF IT IS RUN TWICE IT ADDS DATA.

f = urllib.request.urlopen('http://api.wunderground.com/api/e4026bac17a9a878/forecast10day/q/GA/ATL.json')
json_string = f.read().decode('utf-8')
parsed_json = json.loads(json_string)
for i in range(0,10):
    year = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['year']
    month = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['month']
    day = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['day']
    date = dt.date(year,month,day)
    ForecastH = np.float32(parsed_json['forecast']['simpleforecast']['forecastday'][i]['high']['fahrenheit'])
    ForecastL = np.float32(parsed_json['forecast']['simpleforecast']['forecastday'][i]['low']['fahrenheit'])
    Mean =(ForecastH+ForecastL)/2
 
    Temptable = Temptable.append({'date':date,'todayT':Mean},ignore_index=True) 


#df to_csv('file.csv')
#plot.show()
#print Temptable
print (date, Mean)
Temptable.plot()

##
## The file "ATLtemptbl" is created where the "10 day future mean" temperature
## is calculated through today's date. We get lucky here and we dont
## record any duplicate info that may be shown in CAPS as a programming
## problem above.

ATLtemptbla = pd.DataFrame(columns=['date','todayT','10daymean'])

count = 3650 + numdays

for i in range(0, count):
    date = Temptable ['date'].ix[i]
    todayT = Temptable ['todayT'].ix[i]
    a = np.mean([Temptable['todayT'].ix[(i):(i+10)]])
    ATLtemptbla = ATLtemptbla.append({'date':date,'todayT':todayT,'10daymean':a},ignore_index=True)  

ATLtemptbla['date'] =  pd.to_datetime(ATLtemptbla['date'])    
    
#print ATLtemptbl
ATLtemptbla.plot()

##

url="https://www.google.com/finance/historical?output=csv&q=UNG&startdate=Jan+1+2007"
s=requests.get(url).content
DataPALLonly=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataPALLonly['Date'] =  pd.to_datetime(DataPALLonly['Date'])

url="https://www.google.com/finance/historical?output=csv&q=USO&startdate=Jan+1+2007"
s=requests.get(url).content
DataCARZ=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataCARZ['Date'] =  pd.to_datetime(DataCARZ['Date'])

url="https://www.google.com/finance/historical?output=csv&q=DUK&startdate=Jan+1+2007"
s=requests.get(url).content
DataNILSY=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataNILSY['Date'] =  pd.to_datetime(DataNILSY['Date'])

url="https://www.quandl.com/api/v3/datasets/CHRIS/CME_SZ2.csv?api_key=VM21Mn8fqyqEEsY2Mmxt"
s=requests.get(url).content
Futures=pd.read_csv(io.StringIO(s.decode('utf-8')))
Futures['Date'] =  pd.to_datetime(Futures['Date'])

#print DataNILSY

##

merged_inner = pd.merge(left=DataPALLonly,right=DataCARZ, left_on='Date', right_on='Date')
DataPALL1 = pd.merge(left=merged_inner,right=DataNILSY, left_on='Date', right_on='Date')
DataPALL2 = pd.merge(left=DataPALL1,right=Futures, left_on='Date', right_on='Date')
DataPALL = pd.merge(left=DataPALL2,right=ATLtemptbla, left_on='Date', right_on='date')
DataPALL.columns = ["Date", "Open_x", "High_x", "Low_x","Adj Close_x","Volume_x","8","9","10","Adj Close_y","12","14","15","16","Adj Close","19",
                    "1","2","3","4","5","Settle","6","7","88","todayT","10daymean"]



#print DataPALL

##

training_test_data = pd.DataFrame(
  columns=[
    'return_positive', 
    'return_negative',
    'HighbyClose',
    'LowbyClose',
    'VolumebyAvgVolume',
    '1DayChange',
    '10DayChange',
    'CARZrat',
    'NILSYrat',
    'Settle',
    '10daymean'
  ])

for i in range(11, 800):
  Increase = DataPALL['Adj Close_x'].ix[i-10]/DataPALL['Adj Close_x'].ix[i]
  if Increase < 0.95:
    return_positive = 1
    return_negative = 0
  else:
    return_positive = 0
    return_negative = 1

  
  A_rank = (DataPALL['Volume_x'].ix[i+1],DataPALL['Volume_x'].ix[i+2],
             DataPALL['Volume_x'].ix[i+3],DataPALL['Volume_x'].ix[i+4],DataPALL['Volume_x'].ix[i+5],
             DataPALL['Volume_x'].ix[i+6],DataPALL['Volume_x'].ix[i+7],DataPALL['Volume_x'].ix[i+8],
             DataPALL['Volume_x'].ix[i+9],DataPALL['Volume_x'].ix[i+10],)
  arr = np.array(A_rank)
  AvgVolume = np.mean(arr)
  High = np.float32(DataPALL['High_x'].ix[i])
  Low = np.float32(DataPALL['Low_x'].ix[i])
  Volume = DataPALL['Volume_x'].ix[i]
  Close = DataPALL['Adj Close_x'].ix[i]
  HighbyClose = High/Close
  LowbyClose = Low/Close
  VolumebyAvgVolume = Volume/AvgVolume
  OneDayChange = (Close-DataPALL['Adj Close_x'].ix[i+1])/Close
  TenDayChange = (Close-DataPALL['Adj Close_x'].ix[i+10])/Close
  CARZrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close_y'].ix[i]
  NILSYrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close'].ix[i]
  Settle = DataPALL['Settle'].ix[i]
  tendaymean = DataPALL['10daymean'].ix[i]


  training_test_data = training_test_data.append(
    {
    'return_positive':return_positive,
    'return_negative':return_negative,
    'HighbyClose':HighbyClose,
    'LowbyClose':LowbyClose,
    'VolumebyAvgVolume':VolumebyAvgVolume,
    '1DayChange':OneDayChange,
    '10DayChange':TenDayChange,
    'CARZrat':CARZrat,
    'NILSYrat':NILSYrat,
    'Settle':Settle,
    '10daymean':tendaymean
    },ignore_index=True)

print (DataPALL['High_x'].ix[11], Low, Close, HighbyClose, LowbyClose)
#training_test_data.describe()

##

predictors_tf = training_test_data[training_test_data.columns[2:]]

classes_tf = training_test_data[training_test_data.columns[:2]]

training_set_size = int(len(training_test_data) * 0.95)
test_set_size = len(training_test_data) - training_set_size

training_predictors_tf = predictors_tf[:training_set_size]
training_classes_tf = classes_tf[:training_set_size]
test_predictors_tf = predictors_tf[training_set_size:]
test_classes_tf = classes_tf[training_set_size:]

#print training_predictors_tf
#training_predictors_tf.describe()

##

def tf_confusion_metrics(model, actual_classes, session, feed_dict):
  predictions = tf.argmax(model, 1)
  actuals = tf.argmax(actual_classes, 1)

  ones_like_actuals = tf.ones_like(actuals)
  zeros_like_actuals = tf.zeros_like(actuals)
  ones_like_predictions = tf.ones_like(predictions)
  zeros_like_predictions = tf.zeros_like(predictions)

  tp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  tn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  fp_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, zeros_like_actuals), 
        tf.equal(predictions, ones_like_predictions)
      ), 
      "float"
    )
  )

  fn_op = tf.reduce_sum(
    tf.cast(
      tf.logical_and(
        tf.equal(actuals, ones_like_actuals), 
        tf.equal(predictions, zeros_like_predictions)
      ), 
      "float"
    )
  )

  tp, tn, fp, fn = \
    session.run(
      [tp_op, tn_op, fp_op, fn_op], 
      feed_dict
    )

  tpr = float(tp)/(float(tp) + float(fn))
  fpr = float(fp)/(float(tp) + float(fn))

  accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

  recall = tpr
  precision = float(tp)/(float(tp) + float(fp))
  
  f1_score = (2 * (precision * recall)) / (precision + recall)
  
  print ('Precision = ', precision)
  print ('Recall = ', recall)
  print ('F1 Score = ', f1_score)
  print ('Accuracy = ', accuracy)

##

sess1 = tf.Session()

num_predictors = len(training_predictors_tf.columns)
num_classes = len(training_classes_tf.columns)

feature_data = tf.placeholder("float", [None, num_predictors])
actual_classes = tf.placeholder("float", [None, 2])

weights1 = tf.Variable(tf.truncated_normal([9, 19], stddev=0.0001))
biases1 = tf.Variable(tf.ones([19]))

weights2 = tf.Variable(tf.truncated_normal([19, 10], stddev=0.0001))
biases2 = tf.Variable(tf.ones([10]))
                     
weights3 = tf.Variable(tf.truncated_normal([10, 2], stddev=0.0001))
biases3 = tf.Variable(tf.ones([2]))

hidden_layer_1 = tf.nn.relu(tf.matmul(feature_data, weights1) + biases1)
hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, weights2) + biases2)
model = tf.nn.softmax(tf.matmul(hidden_layer_2, weights3) + biases3)

cost = -tf.reduce_sum(actual_classes*tf.log(model))

train_op1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

init = tf.global_variables_initializer()
sess1.run(init)

##

correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(actual_classes, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for i in range(1, 30001):
  sess1.run(
    train_op1, 
    feed_dict={
      feature_data: training_predictors_tf.values, 
      actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
    }
  )
  if i%5000 == 0:
    print (i, sess1.run(
      accuracy,
      feed_dict={
        feature_data: training_predictors_tf.values, 
        actual_classes: training_classes_tf.values.reshape(len(training_classes_tf.values), 2)
      }
    )
           )
##

feed_dict= {
  feature_data: test_predictors_tf.values,
  actual_classes: test_classes_tf.values.reshape(len(test_classes_tf.values), 2)
}

tf_confusion_metrics(model, actual_classes, sess1, feed_dict)

classification = sess1.run(tf.argmax(model, 1), feed_dict)
print (test_predictors_tf)
print (classification)

##

beg1 = pd.DataFrame(
  columns=[
    'date', 
    'todayT'])

dt.datetime.today().strftime("%m/%d/%Y")
day = np.int(dt.datetime.today().strftime("%d"))
month = np.int(dt.datetime.today().strftime("%m"))
numdays = np.int(dt.datetime.today().strftime("%j"))-1


url="https://www.wunderground.com/history/airport/ATL/2017/1/1/CustomHistory.html?dayend={}&monthend={}&yearend=2017&format=1&_ga=2.157831161.2102240471.1496308854-871259340.1496277779".format(day,month)
s=requests.get(url).content
ATL1=pd.read_csv(io.StringIO(s.decode('utf-8')))
ATL = ATL1[["EDT", "Mean TemperatureF"]]

f = urllib.request.urlopen('http://api.wunderground.com/api/e4026bac17a9a878/forecast10day/q/GA/ATL.json')
json_string = f.read().decode('utf-8')
parsed_json = json.loads(json_string)

for i in range(0, numdays):
    date = ATL ['EDT'].ix[i]
    todayT = ATL ['Mean TemperatureF'].ix[i]
#    a = np.mean([ATL['Mean TemperatureF'].ix[(i):(i+10)]])
    beg1 = beg1.append({'date':date,'todayT':todayT
                                  #,'AvgTATL':a
                                 },ignore_index=True)  

#print beg1
beg1.plot()

##

f = urllib.request.urlopen('http://api.wunderground.com/api/e4026bac17a9a878/forecast10day/q/GA/ATL.json')
json_string = f.read().decode('utf-8')
parsed_json = json.loads(json_string)
for i in range(0,10):
    year = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['year']
    month = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['month']
    day = parsed_json['forecast']['simpleforecast']['forecastday'][i]['date']['day']
    date = dt.date(year,month,day)
    ForecastH = np.float32(parsed_json['forecast']['simpleforecast']['forecastday'][i]['high']['fahrenheit'])
    ForecastL = np.float32(parsed_json['forecast']['simpleforecast']['forecastday'][i]['low']['fahrenheit'])
    Mean =(ForecastH+ForecastL)/2
 
    beg1 = beg1.append({'date':date,'todayT':Mean
                                  #,'AvgTATL':a
                                 },ignore_index=True) 


#df to_csv('file.csv')
#beg1['todayT'].plot()
#plot.show()
#print beg1
beg1.plot()

##

beg2 = pd.DataFrame(columns=['date','todayT','10daymean'])

for i in range(0, numdays
              ):
    date = beg1 ['date'].ix[i]
    todayT = beg1 ['todayT'].ix[i]
    a = np.mean([beg1['todayT'].ix[(i):(i+10)]])
    beg2 = beg2.append({'date':date,'todayT':todayT,'10daymean':a},ignore_index=True)  

beg2['date'] =  pd.to_datetime(beg2['date'])    
    
#print beg2
beg2.plot()

##

url="https://www.google.com/finance/historical?output=csv&q=UNG&startdate=Jan+1+2007"
s=requests.get(url).content
DataPALLonly=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataPALLonly['Date'] =  pd.to_datetime(DataPALLonly['Date'])

url="https://www.google.com/finance/historical?output=csv&q=USO&startdate=Jan+1+2007"
s=requests.get(url).content
DataCARZ=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataCARZ['Date'] =  pd.to_datetime(DataCARZ['Date'])

url="https://www.google.com/finance/historical?output=csv&q=DUK&startdate=Jan+1+2007"
s=requests.get(url).content
DataNILSY=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataNILSY['Date'] =  pd.to_datetime(DataNILSY['Date'])

url="https://www.quandl.com/api/v3/datasets/CHRIS/CME_SZ2.csv?api_key=VM21Mn8fqyqEEsY2Mmxt"
s=requests.get(url).content
Futures=pd.read_csv(io.StringIO(s.decode('utf-8')))
Futures['Date'] =  pd.to_datetime(Futures['Date'])

#print DataNILSY

##

merged_inner = pd.merge(left=DataPALLonly,right=DataCARZ, left_on='Date', right_on='Date')
DataPALL1 = pd.merge(left=merged_inner,right=DataNILSY, left_on='Date', right_on='Date')
DataPALL2 = pd.merge(left=DataPALL1,right=Futures, left_on='Date', right_on='Date')
DataPALL = pd.merge(left=DataPALL2,right=beg2, left_on='Date', right_on='date')
#DataPALL.columns = ["Date", "Open_x", "High_x", "Low_x","Close_x","Volume_x","Adj Close_x","8","9","10","11","12","Adj Close_y","14","15","16","17","18","Adj Close","20","21","22","23","24","Settle","26","27"]
DataPALL.columns = ["Date", "Open_x", "High_x", "Low_x","Adj Close_x","Volume_x","8","9","10","Adj Close_y","12","14","15","16","Adj Close","19",
                    "1","2","3","4","5","Settle","6","7","88","todayT","10daymean"]



print (DataPALL)

##

beginingofyear = pd.DataFrame(
  columns=[
    #'return_positive', 
    #'return_negative',
    'HighbyClose',
    'LowbyClose',
    'VolumebyAvgVolume',
    '1DayChange',
    '10DayChange',
    #'AvgVolume','High','Low','Volume','Close',
    'CARZrat',
    'NILSYrat','Settle','10daymean'
  ])

for i in range(0, 100):
  
  A_rank = (DataPALL['Volume_x'].ix[i+1],DataPALL['Volume_x'].ix[i+2],
             DataPALL['Volume_x'].ix[i+3],DataPALL['Volume_x'].ix[i+4],DataPALL['Volume_x'].ix[i+5],
             DataPALL['Volume_x'].ix[i+6],DataPALL['Volume_x'].ix[i+7],DataPALL['Volume_x'].ix[i+8],
             DataPALL['Volume_x'].ix[i+9],DataPALL['Volume_x'].ix[i+10],)
  arr = np.array(A_rank)
  AvgVolume = np.mean(arr)
  High = np.float32(DataPALL['High_x'].ix[i])
  Low = np.float32(DataPALL['Low_x'].ix[i])
  Volume = DataPALL['Volume_x'].ix[i]
  Close = DataPALL['Adj Close_x'].ix[i]
  HighbyClose = High/Close
  LowbyClose = Low/Close
  VolumebyAvgVolume = Volume/AvgVolume
  OneDayChange = (Close-DataPALL['Adj Close_x'].ix[i+1])/Close
  TenDayChange = (Close-DataPALL['Adj Close_x'].ix[i+10])/Close
  CARZrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close_y'].ix[i]
  NILSYrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close'].ix[i]
  Settle = DataPALL['Settle'].ix[i]/DataPALL['Adj Close_x'].ix[i]
  tendaymean = DataPALL['10daymean'].ix[i]

  #SELECT stock, STD(Close-Price) from `historic_prices` where stock = "NFLX" AND date > "2010-01-01" group by stock


  beginingofyear = beginingofyear.append(

    {
    #'return_positive':return_positive,
    #'return_negative':return_negative,
    'HighbyClose':HighbyClose,
    'LowbyClose':LowbyClose,
    'VolumebyAvgVolume':VolumebyAvgVolume,
    '1DayChange':OneDayChange,
    '10DayChange':TenDayChange,
    #'AvgVolume':AvgVolume,
    #'High':High,
    #'Low':Low,
    #'Volume':Volume,
    #'Close':Close,
    'CARZrat':CARZrat,
    'NILSYrat':NILSYrat,
    'Settle':Settle,
    '10daymean':tendaymean
   
    },
    
    ignore_index=True)

print (beginingofyear)

##

#feed_dict= {feature_data: test_predictors_tf.values}
feed_dict= {feature_data: beginingofyear}
classification = sess1.run(tf.argmax(model, 1), feed_dict)

print (feed_dict)
print (classification[0])
#classification.tolist()
#print 'NN predicted'
#print beginingofyear
#print test_classes_tf
print (classification)

##

output = pd.DataFrame(columns=['date','classifier','price','10daymean'])

date = dt.datetime.today().strftime("%m/%d/%Y")

for i in range(0,100):

    
#    date = dt.datetime.today() - dt.timedelta(days=i)
    date = DataPALL['Date'].ix[i]
    High = np.float32(DataPALL['High_x'].ix[i])
    Low = np.float32(DataPALL['Low_x'].ix[i])
    Volume = DataPALL['Volume_x'].ix[i]
    Close = DataPALL['Adj Close_x'].ix[i]
    HighbyClose = High/Close
    LowbyClose = Low/Close
    VolumebyAvgVolume = Volume/AvgVolume
    OneDayChange = (Close-DataPALL['Adj Close_x'].ix[i+1])/Close
    TenDayChange = (Close-DataPALL['Adj Close_x'].ix[i+10])/Close
    CARZrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close_y'].ix[i]
    NILSYrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close'].ix[i]
    Settle = DataPALL['Settle'].ix[i]
    tendaymean = DataPALL['10daymean'].ix[i]
    classifier = classification[i]
    output = output.append({'date':date,
                            'classifier':classifier,'price':Close,'10daymean':tendaymean
                           },ignore_index=True) 


#output.to_csv('file.csv')
#beg1['todayT'].plot()
#plot.show()
print (output)
#beg1.plot()

##

#This will be the code more for intraday trading
# Under construction

url="http://finance.yahoo.com/d/quotes.csv?s=MSFT+PALL+NILSY&f=pc1ghv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
c.columns = ["Pbefore", "Change", "Low", "High","Volume"]

print (c)


PALLClose = c['Pbefore'].ix[0] + c['Change'].ix[0]
#PALLHigh =


HighbyClose = High/Close
LowbyClose = Low/Close
VolumebyAvgVolume = Volume/AvgVolume
OneDayChange = (Close-DataPALL['Adj Close_x'].ix[i+1])/Close
TenDayChange = (Close-DataPALL['Adj Close_x'].ix[i+10])/Close
CARZrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close_y'].ix[i]
NILSYrat = DataPALL['Adj Close_x'].ix[i]/DataPALL['Adj Close'].ix[i]

#Pall1 = c[[0],['Pbefore']]
#Pall2 = c.loc[[0],['Change']]
#close = Pall1+Pall2
print (Volume)

#d = 3 + c

##

url="https://www.google.com/finance/historical?output=csv&q=UNG"
s=requests.get(url).content
DataPALLonly=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataPALLonly['Date'] =  pd.to_datetime(DataPALLonly['Date'])

url="https://www.google.com/finance/historical?output=csv&q=USO"
s=requests.get(url).content
DataCARZ=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataCARZ['Date'] =  pd.to_datetime(DataCARZ['Date'])

url="https://www.google.com/finance/historical?output=csv&q=DUK"
s=requests.get(url).content
DataNILSY=pd.read_csv(io.StringIO(s.decode('utf-8')))
DataNILSY['Date'] =  pd.to_datetime(DataNILSY['Date'])

url="https://www.quandl.com/api/v3/datasets/CHRIS/CME_SZ2.csv?api_key=VM21Mn8fqyqEEsY2Mmxt"
s=requests.get(url).content
Futures=pd.read_csv(io.StringIO(s.decode('utf-8')))
Futures['Date'] =  pd.to_datetime(Futures['Date'])

#print DataPALLonly

##

merged_inner = pd.merge(left=DataPALLonly,right=DataCARZ, left_on='Date', right_on='Date')
DataPALL1 = pd.merge(left=merged_inner,right=DataNILSY, left_on='Date', right_on='Date')
DataPALL = pd.merge(left=DataPALL1,right=Futures, left_on='Date', right_on='Date')
#DataPALL.columns = ["Date", "Open_x", "High_x", "Low_x","Close_x","Volume_x","Adj Close_x","8","9","10","11","12","Adj Close_y","14","15","16","17","18","Adj Close","20","21","22","23","24","Settle","26","27"]
DataPALL.columns = ["Date", "Open_x", "High_x", "Low_x","Adj Close_x","Volume_x","8","9","10","Adj Close_y","12","14","15","16","Adj Close","19","20","21","22","23","24","Settle","25","26"]



#print DataPALL

##

A_rank = (DataPALL['Volume_x'].ix[0],DataPALL['Volume_x'].ix[1],
             DataPALL['Volume_x'].ix[2],DataPALL['Volume_x'].ix[3],DataPALL['Volume_x'].ix[4],
             DataPALL['Volume_x'].ix[5],DataPALL['Volume_x'].ix[6],DataPALL['Volume_x'].ix[7],
             DataPALL['Volume_x'].ix[8],DataPALL['Volume_x'].ix[9],)
arr = np.array(A_rank)
AvgVolume = np.mean(arr)
High = DataPALL['High_x'].ix[0]
Low = DataPALL['Low_x'].ix[0]
Volume = DataPALL['Volume_x'].ix[0]
Close = DataPALL['Adj Close_x'].ix[0]
HighbyClose = High/Close
LowbyClose = Low/Close
VolumebyAvgVolume = Volume/AvgVolume
OneDayChange = (Close-DataPALL['Adj Close_x'].ix[1])/Close
TenDayChange = (Close-DataPALL['Adj Close_x'].ix[10])/Close
CARZrat = DataPALL['Adj Close_x'].ix[0]/DataPALL['Adj Close_y'].ix[0]
NILSYrat = DataPALL['Adj Close_x'].ix[0]/DataPALL['Adj Close'].ix[0]
Settle = DataPALL['Settle'].ix[0]/DataPALL['Adj Close_x'].ix[0]

print (Close)

##

feed_dict= {feature_data: [[HighbyClose,LowbyClose,VolumebyAvgVolume,OneDayChange,TenDayChange,CARZrat,NILSYrat,Settle] ]}
classification = sess1.run(tf.argmax(model, 1), feed_dict)

print (HighbyClose,LowbyClose,VolumebyAvgVolume,OneDayChange,TenDayChange,CARZrat,NILSYrat,Settle)
print ("Natural Gas Down: ", classification)
