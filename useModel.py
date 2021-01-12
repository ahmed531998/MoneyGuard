import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from datetime import datetime
from weka.attribute_selection import ASSearch, ASEvaluation
import weka.core.serialization as sr
from weka.core.converters import Loader
from weka.filters import Filter
import weka.core.dataset as ds
import numpy as np
import os
import threading

def loadModel(modelPath):
    model = Classifier(jobject=sr.read(modelPath))
    return model

def loadFilter(filterPath):
    filter = Filter(jobject=sr.read(filterPath))
    return filter


def stringToInstance(string):
    lock = threading.Lock()
    lock.acquire()
    with open('step.arff', 'a') as f:
        f.write('\n')
        f.write(string)

    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file("step.arff")

    with open('step.arff', "r+", encoding="utf-8") as file:
        file.seek(0, os.SEEK_END)
        pos = file.tell() - 1
        while pos > 0:
            if file.read(1) == "\n":
                break
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        if pos > 0:
            file.seek(pos, os.SEEK_SET)
            file.truncate()
    lock.release()
    return data


def appendToDataSet(instance):
    lock = threading.Lock()
    lock.acquire()
    with open('dataSources/fraud.arff', 'a') as f:
        f.write('\n')
        f.write(instance)
    lock.release()

def addFraud(string):
    lock = threading.Lock()
    lock.acquire()
    with open('fraudTrans.txt', 'a') as f:
        f.write('\n')
        f.write(string)
    lock.release()

def preprocess(data):
    data.class_is_last()
    discretizer = loadFilter('filters/discretizer')
    stn = loadFilter('filters/stn')
    remover = loadFilter('filters/remover')
    # discretize age
    discData = discretizer.filter(data)
    convData = stn.filter(discData)
    newData = remover.filter(convData)
    return newData


def stream(file, option):
    jvm.start(packages=True)

    if option == 1:
        print("Hi! This is a protected command, please insert the password to proceed!")
        for x in range(3):
            password = input('')
            if password.strip() == 'DMMLproject':
                print("All good!")
                break
            else:
                if x == 2:
                    print(
                        "This command is protected and can be used only by an administrator, please use another command.")
                    return
                else:
                    print("Wrong password, please provide the correct password")

    hoeffding = loadModel('models/HoeffdingTree.model')
    f = open(file, 'r')
    while True:
        line = f.readline()
        if not line:
            break
        if option == 0:
            classifyOne(line.strip(), hoeffding)
        else:
            print('Stream update start at: ', datetime.now().time())
            hoeffding = retrainOneInc(line.strip(), hoeffding)
            print('Stream update end at: ', datetime.now().time())
    f.close()
    sr.write('models/HoeffdingTree.model', hoeffding)


def retrainOneInc(string, classifier):
    data = stringToInstance(string)
    preProcessedData = preprocess(data)
    classifier.update_classifier(preProcessedData.get_instance(0))
    appendToDataSet(string)
    return classifier

def classifyOne(string=None,classifier=None):
    if not classifier:
        jvm.start(packages=True)
        classifier = loadModel('models/randomForest.model')

    if not string:
        print('Copy your transaction informations here! Please, use a comma separated list!')
        string = input('')

    data = stringToInstance(string)
    preProcessedData = preprocess(data)
    result = classifier.classify_instance(preProcessedData.get_instance(0))
    final = int(result)
    toUpdate = string[:-1]
    toUpdate += str(final)
    appendToDataSet(toUpdate)
    if final == 0:
        print("The transaction is safe!")
    else:
        print("ATTENTION!\n The transaction seems to be a scam, contact your bank and let them know!")
        addFraud(toUpdate)
