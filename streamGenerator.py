import pandas
from datetime import datetime, date
import time
import random
import string
import numpy

def randStr(chars = string.ascii_lowercase + string.digits, N=32):
	return ''.join(random.choice(chars) for _ in range(N))

def dobTOage(ref, x):
    birthdate = datetime.strptime(x, '%Y-%m-%d')
    now = datetime.strptime(ref, '%Y-%m-%d %H:%M:%S')
    return now.year - birthdate.year - ((now.month, now.day) < (birthdate.month, birthdate.day))

def dateTOunix(x):
    return time.mktime(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())

def getDataEntity(dataList):
    List = []
    for i in range (0, len(dataList)):
        List.append(dataList[i])
    return List

def getIndex(maxVal):
    return numpy.random.randint(maxVal)

def getListOfTrans(dataset, distinctValuesTrain, numTrans):
    fraud = True
    fraudCnt = 0

    df = pandas.DataFrame(columns=['trans_date_trans_time', 'unix_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
                   'lat', 'long', 'city_pop', 'job', 'age', 'dob', 'trans_num',
                   'merch_lat', 'merch_long' , 'amt', 'is_fraud'])
    
    for i in range (0, numTrans):
        d={}
        d['trans_date_trans_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        d['unix_time'] = dateTOunix(d['trans_date_trans_time'])
        d['cc_num'] = distinctValuesTrain['cc_num'][getIndex(len(distinctValuesTrain['cc_num']))]
        userDF = dataset[dataset['cc_num'] == d['cc_num']]
        d['first'] = userDF.iloc[0]['first']
        d['last'] = userDF.iloc[0]['last']
        d['gender'] = userDF.iloc[0]['gender']
        d['street'] = userDF.iloc[0]['street']
        d['city'] = userDF.iloc[0]['city']
        d['state'] = userDF.iloc[0]['state']
        d['zip'] = userDF.iloc[0]['zip']
        d['lat'] = userDF.iloc[0]['lat']
        d['long'] = userDF.iloc[0]['long']
        d['city_pop'] = userDF.iloc[0]['city_pop']
        d['job'] = userDF.iloc[0]['job']
        d['dob'] = distinctValuesTrain['dob'][getIndex(len(distinctValuesTrain['dob']))]
        d['age'] = dobTOage(d['trans_date_trans_time'], d['dob'])
        d['merchant'] = distinctValuesTrain['merchant'][getIndex(len(distinctValuesTrain['merchant']))]
        userDF = dataset[dataset['merchant'] == d['merchant']]
        d['merch_long'] = userDF.iloc[0]['merch_long']
        d['merch_lat'] = userDF.iloc[0]['merch_lat']  
        d['category'] = distinctValuesTrain['category'][getIndex(len(distinctValuesTrain['category']))] 
        d['trans_num'] = randStr()
        d['amt'] = distinctValuesTrain['amt'][getIndex(len(distinctValuesTrain['amt']))]
        d['is_fraud'] = 1 if fraud else 0
        if fraud:
            fraudCnt +=1
            if fraudCnt >= 0.005 * numTrans:
                fraud = False
        x = {'trans_date_trans_time': [d['trans_date_trans_time']], 'unix_time': [d['unix_time']],
             'cc_num': [d['cc_num']], 'first': [d['first']], 'last': [d['last']], 'gender': [d['gender']], 'street': [d['street']], 'city': [d['city']],
             'state': [d['state']], 'zip': [d['zip']], 'lat': [d['lat']], 'long': [d['long']], 'city_pop': [d['city_pop']], 'job': [d['job']],
             'dob': [d['dob']], 'age': [d['age']], 'merchant': [d['merchant']], 'merch_long': [d['merch_long']], 'merch_lat': [d['merch_lat']],
             'category': [d['category']], 'trans_num': [d['trans_num']], 'amt': [d['amt']], 'is_fraud': [d['is_fraud']]}
        df1 = pandas.DataFrame(data=x)
        df = df.append(df1)
    return df

def writeLine(file, values, lineEndChar, data=False):
    for j in range(0, len(values)):
        w = str(values[j])
        if data:
            if j == 1 or j == 17:
                file.write('"' + w +'",')
                continue
        if (' ' in w):
            if ("'" in w):
                w = '"'+w+'"'
            else:
                w = "'"+w+"'"
        else:
            if ("'" in w):
                w = '"'+w+'"'        
        if j == len(values)-1:
            file.write(w +lineEndChar+"\n")
        else:
            file.write(w +",")
        
        
def writeToFile(dataList, fileName):
    f = open(fileName, "w")    
    for datum in dataList:
        writeLine(f, datum, '', data=True)
    f.close()



numInstances = 1000

dataset = pandas.read_csv('data.csv')

attributeListTrain = dataset.columns.tolist()

distinctValuesTrain = {}
for attribute in attributeListTrain:
    distinctValuesTrain[attribute] = pandas.unique(dataset[attribute].values.ravel())

df = getListOfTrans(dataset, distinctValuesTrain, numInstances)
df = df.sample(frac=1).reset_index(drop=True)
data = getDataEntity(df.values.tolist())

writeToFile(data, 'stream.txt')
