import pandas
from datetime import datetime, date
import time

def dobTOage(ref, x):
    birthdate = datetime.strptime(x, '%Y-%m-%d')
    now = datetime.strptime(ref, '%Y-%m-%d %H:%M:%S')
    return now.year - birthdate.year - ((now.month, now.day) < (birthdate.month, birthdate.day))

def dateTOunix(x):
    return time.mktime(datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())


def getAttributeEntity(attributeList, typeList, dateFormatList):
    attDict = []
    for i in range (0, len(attributeList)):
        dic = {}
        dic['name'] = attributeList[i]
        dic['type'] = typeList[i]
        if dic['type'] == 'DATE':
            dic['dateFormat'] = dateFormatList[i]
        attDict.append(dic)
    return attDict

def getDataEntity(dataList):
    dataDict = []
    for i in range (0, len(dataList)):
        dic = {}
        dic['values'] = dataList[i]
        dataDict.append(dic)
    return dataDict

def createDict(df, relationName, attributeList, typeList, dateFormatList):
    rec = {}
    attributes = getAttributeEntity(attributeList, typeList, dateFormatList)
    data = getDataEntity(df.values.tolist())
    rec ['header'] = {
        'relation': relationName,
        'attributes':  attributes}
    rec ['data'] = data
    return rec

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
        
        
def writeToArff(dic, fileName, nomValues):
    f = open(fileName, "w")
    f.write("@RELATION " + dic['header']['relation'] + '\n')
    f.write('\n')
    header = dic['header']
    attributes = header['attributes']
    
    for i in range(0, len(attributes)):
        attribute = attributes[i]
        if(attribute['type'] == 'nominal'):
            f.write("@ATTRIBUTE " + attribute['name'] + ' {')
            writeLine(f, nomValues[attribute['name']], '}')
        elif (attribute['type'] == 'DATE'):
            f.write("@ATTRIBUTE " + attribute['name'] + ' ' + attribute['type'] + ' "' + attribute['dateFormat'] + '"\n')
        else:
            f.write("@ATTRIBUTE " + attribute['name'] + ' ' + attribute['type'] + '\n')
    f.write('\n')
    f.write("@DATA\n")
    for datum in dic['data']:
        values = datum['values']
        writeLine(f, values, '', data=True)
    f.close()



def filterDF(df, col, pattern):
    filterT = df[col].str.contains(pattern)
    df = df[~filterT]
    return df


trainDF = pandas.read_csv('dataSources/fraudTrain.csv')
testDF = pandas.read_csv('dataSources/fraudTest.csv')

trainDF = filterDF(trainDF, 'trans_date_trans_time', "2019-03-31 02:")
testDF = filterDF(testDF, 'trans_date_trans_time', "2019-03-31 02:")

trainDF = trainDF.append(testDF,ignore_index=True)

trainDF['age'] = trainDF.apply(lambda x: dobTOage(x['trans_date_trans_time'], x['dob']), axis=1)
trainDF['unix_time'] = trainDF.apply(lambda x: dateTOunix(x['trans_date_trans_time']), axis=1)


dataset = trainDF[['trans_date_trans_time', 'unix_time', 'cc_num', 'merchant', 'category',
                   'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
                   'lat', 'long', 'city_pop', 'job', 'age', 'dob', 'trans_num',
                   'merch_lat', 'merch_long' , 'amt', 'is_fraud']]

dataset.to_csv('dataSources/data.csv')
#trainDF.drop(['id', 'cc_num', 'first', 'last', 'trans_num', 'unix_time', 'zip', 'state', 'dob', 'merchant', 'street', 'city'], axis='columns', inplace=True)

attributeListTrain = dataset.columns.tolist()


distinctValuesTrain = {}
for attribute in attributeListTrain:
    distinctValuesTrain[attribute] = pandas.unique(dataset[attribute].values.ravel())

typeList = ['DATE', 'NUMERIC', 'STRING', 'STRING', 'nominal', 'STRING', 'STRING', 'nominal', 'STRING', 'STRING', 'STRING',
            'STRING', 'NUMERIC', 'NUMERIC', 'NUMERIC', 'nominal', 'NUMERIC', 'DATE', 'STRING', 'NUMERIC', 'NUMERIC', 'NUMERIC', 'nominal']

dateFormatList = ['yyyy-MM-dd HH:mm:ss']*23
dateFormatList[17] = 'yyyy-MM-dd'


dicTrain = createDict(dataset, 'fraud', attributeListTrain, typeList, dateFormatList)

writeToArff(dicTrain, 'dataSources/fraud.arff', distinctValuesTrain)
