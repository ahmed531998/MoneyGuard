import matplotlib.pyplot as plt
import pandas

def plotHistogram(col):
    plt.hist(col, color='blue', alpha=0.5)
    plt.title("Histogram of '{var_name}'".format(var_name=col.name))
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.show()

def plotHistogramCateg(col, label):
    plt.hist(list(col[label==1]), color='red', alpha=0.5, label='Fraud')
    plt.hist(list(col[label==0]), color='blue', alpha=0.5, label='Not Fraud')
    plt.figure(figsize=(150, 100))

    plt.title("Histogram of '{var_name}' by class".format(var_name=col.name))
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.legend(loc='upper right')
    plt.show()

def studyCorrelation(att, cls):
    plt.scatter(att, cls, color='blue')
    plt.figure(figsize=(150, 100))
    #plt.title("Correlation Analysis of '{var_name}' with class".format(var_name=att.name))
    plt.xlabel('amt')
    plt.ylabel('class')
    plt.legend(loc='upper right')
    plt.show()

def plotLine(col):
    l = list(col)
    l.sort()
    index = [i for i in range(0, len(l))]
    plt.scatter(index, l)

    plt.figure(figsize=(150, 100))
    plt.title("Distribution of '{var_name}'".format(var_name=col.name))
    plt.xlabel('index')
    plt.ylabel('value')
    plt.legend(loc='upper right')
    plt.show()

def plotKDE(col, cls):
    fraud = col[cls==1]
    safe = col[cls==0]

    df = pandas.DataFrame({
        'fraud': fraud,
        'safe': safe,}
    )
    df.plot.kde()
    plt.show()

def plotBar(dataset,col):
    vals1 = dataset[col][dataset['is_fraud']==1].value_counts().sort_values()
    vals2 = dataset[col][dataset['is_fraud']==0].value_counts().sort_values()

    vals1.plot(kind="barh", fontsize=8, color='red')
    vals2.plot(kind="barh", fontsize=8, color='blue')
    plt.figure(figsize=(300, 300))

    plt.show()

dataset = pandas.read_csv('dataSources/data.csv')

#get general information
print(dataset.info())
print(dataset.describe())

# study nominal values
print(dataset['first'].value_counts())
print(dataset['last'].value_counts())
print(dataset['gender'].value_counts())
print(dataset['merchant'].value_counts())
print(dataset['category'].value_counts())
print(dataset['street'].value_counts())
print(dataset['city'].value_counts())
print(dataset['state'].value_counts())
print(dataset['zip'].value_counts())
print(dataset['job'].value_counts())
print(dataset['is_fraud'].value_counts())
plotBar(dataset, 'gender')
plotBar(dataset, 'category')
plotHistogram(dataset['is_fraud'])


#study continuous attributes
plotKDE(dataset['amt'], dataset['is_fraud'])
plotKDE(dataset['lat'], dataset['is_fraud'])
plotKDE(dataset['long'], dataset['is_fraud'])
plotKDE(dataset['merch_lat'], dataset['is_fraud'])
plotKDE(dataset['merch_long'], dataset['is_fraud'])
plotKDE(dataset['age'], dataset['is_fraud'])
plotKDE(dataset['city_pop'], dataset['is_fraud'])
plotKDE(dataset['unix_time'], dataset['is_fraud'])
