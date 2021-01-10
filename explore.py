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
    fraud = att[cls==1]
    safe = att[cls==0]

    fraudVals = fraud.value_counts()
    safeVals = safe.value_counts()

    plt.scatter(fraudVals.index, fraudVals.values, color='red')
    plt.scatter(safeVals.index, safeVals.values, color='blue')
    plt.figure(figsize=(150, 100))
    plt.title("Correlation Analysis of '{var_name}' with class".format(var_name=att.name))
    plt.xlabel(att.name)
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

def plotBar(dataset,col):
    vals = dataset[col].value_counts().sort_values()
    vals.plot(kind="barh", fontsize=8)
    plt.figure(figsize=(300, 300))

    plt.show()

dataset = pandas.read_csv('dataSources/data.csv')


#study distribution of numeric
plotHistogramCateg(dataset['merch_lat'], dataset['is_fraud'])
plotHistogramCateg(dataset['merch_long'], dataset['is_fraud'])
plotHistogramCateg(dataset['age'], dataset['is_fraud'])
plotHistogramCateg(dataset['city_pop'], dataset['is_fraud'])
plotHistogramCateg(dataset['lat'], dataset['is_fraud'])
plotHistogramCateg(dataset['long'], dataset['is_fraud'])

#study nominal
plotBar(dataset, 'merchant')
plotBar(dataset,'category')
plotBar(dataset, 'first')
plotBar(dataset,'last')
plotBar(dataset,'gender')
plotBar(dataset,'street')
plotBar(dataset,'city')
plotBar(dataset,'state')
plotBar(dataset,'zip')
plotBar(dataset,'job')

