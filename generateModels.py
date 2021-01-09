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


# percentage = total new/ total old
def undersample(data, percentage):
    if (percentage >= 100):
        return None
    sampler = Filter(classname='weka.filters.supervised.instance.Resample',
                     options=["-B", "1.0", "-S", "1", "-Z", str(percentage)])
    sampler.inputformat(data)
    newData = sampler.filter(data)
    return newData


# percentage = created minority/total old minority
def smote(data, percentage):
    sampler = Filter(classname='weka.filters.supervised.instance.SMOTE',
                     options=["-C", "0", "-K", "5", "-P", str(percentage), "-S", "1"])
    sampler.inputformat(data)
    newData = sampler.filter(data)
    return newData


def resample(data, smotePerc, underPerc, over, under):
    newData = None
    if over and under:
        x = smote(data, smotePerc)
        newData = undersample(x, underPerc)
    elif over:
        newData = smote(data, smotePerc)
    elif under:
        newData = undersample(data, underPerc)
    else:
        newData = None

    return newData


def discretize(data, k, index, file):
    discretizer = Filter(classname='weka.filters.unsupervised.attribute.Discretize',
                         options=["-Y", "-B", str(k), "-M", "-1.0", "-R", str(index), "-precision", "6"])
    discretizer.inputformat(data)
    newData = discretizer.filter(data)
    discretizer.serialize(file)
    return newData


def remove(data, file):
    remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", "1, 18, 19"])
    remove.inputformat(data)
    newData = remove.filter(data)
    remove.serialize(file)
    return newData


def stringToNominal(data, file):
    stn = Filter(classname="weka.filters.unsupervised.attribute.StringToNominal",
                 options=["-R", "2, 3, 5, 6, 8, 9, 10, 11"])
    stn.inputformat(data)
    newData = stn.filter(data)
    stn.serialize(file)
    return newData


def preprocess(data):
    data.class_is_last()

    # discretize age
    discData = discretize(data, 3, 16, 'discretizer')

    # remove irrelevant/redundant attributes
    newData = remove(discData, 'remover')

    # convert string attributes to nominal
    convData = stringToNominal(newData, 'stn')

    # resample data
    finalData = resample(convData, 14622.11999, 85.835, True, True)

    return finalData


def displayResults(evaluationMethod, evaluator):
    print(evaluationMethod)
    print(evaluator.summary(complexity=True))
    print(evaluator.class_details(title='Details'))
    print(evaluator.confusion_matrix)


def classify(data, classifier, cv, modelPath, folds=10, splitPerc=70, randomSeed=10):
    # cross validate the model
    if cv:
        print('CV start at: ', datetime.now().time())
        evaluation = Evaluation(data)
        evaluation.crossvalidate_model(classifier, data, folds, Random(randomSeed))
        print('CV end at: ', datetime.now().time())
        displayResults("Cross Validation", evaluation)

    else:
        # split data into train and test
        print('Split start training at: ', datetime.now().time())
        train, test = data.train_test_split(splitPerc, Random(randomSeed))
        # build classifier with training set
        classifier.build_classifier(train)
        print('Split end training at: ', datetime.now().time())
        evaluation = Evaluation(train)

        print('Split start at: ', datetime.now().time())
        evaluation.test_model(classifier, test)
        print('Split end at: ', datetime.now().time())

        # evaluation.evaluate_model(classifier, ["-t", test])

        displayResults("TrainTestSplit", evaluation)
        sr.write(modelPath, classifier)

def main():
    jvm.start(packages=True, max_heap_size="4g")

    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file("fraud.arff")
    print("Before Preprocessing: \n")
    print(data.attribute_stats(22))
    preProcessedData = preprocess(data)
    print("After Preprocessing: \n")
    print(preProcessedData.attribute_stats(preProcessedData.class_index))

    #setup classifier with attribute selection
    classifier = Classifier(classname="weka.classifiers.meta.AttributeSelectedClassifier")
    aseval = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
    assearch = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])

    classifier.set_property("evaluator", aseval.jobject)
    classifier.set_property("search", assearch.jobject)

    base1 = Classifier(classname="weka.classifiers.bayes.NaiveBayes")
    base2 = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-P", "70", "-I", "30", "-num-slots", "1", "-K", "0", "-M", "1.0",
                                                                                 "-S", "1", "-depth", "50"])
    base3 = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.25", "-M", "2"])
    base4 = Classifier(classname="weka.classifiers.trees.J48", options=["-U", "-M", "2"])
    base5 = Classifier(classname="weka.classifiers.trees.HoeffdingTree", options=["-L", "2", "-S", "1", "-E", "1.0E7", "-H", "0.05", "-M", "0.01",
                                                                                  "-G", "200.0", "-N", "0.0"])
    base6 = Classifier(classname="weka.classifiers.lazy.IBk", options=['-K', '1', '-W', '0'])
    base7 = Classifier(classname="weka.classifiers.bayes.BayesNet")

    #naive bayes - cross validate - traintestSplit
    #print("----------NaiveBayes----------")
    #classifier.set_property("classifier", base1.jobject)
    #classify(preProcessedData,classifier,True,'naiveBayes.model',splitPerc=70,randomSeed=10)
    #classify(preProcessedData,classifier,False,'naiveBayes.model',splitPerc=70,randomSeed=10)

    #random forest - cross validate - traintestSplit
    print("----------RandomForest----------")
    classifier.set_property("classifier", base2.jobject)
    #classify(preProcessedData,classifier,True,'randomForest.model',splitPerc=70,randomSeed=10)
    classify(preProcessedData,classifier,False,'randomForest.model',splitPerc=70,randomSeed=10)

    #decision tree (with pruning) - cross validate - traintestSplit
    #print("----------DecisionTree----------")
    #classifier.set_property("classifier", base3.jobject)
    #classify(preProcessedData,classifier,True,'prunedJ48.model',splitPerc=70,randomSeed=10)
    #classify(preProcessedData,classifier,False,'prunedJ48.model',splitPerc=70,randomSeed=10)

    #decision tree (without pruning) - cross validate - traintestSplit
    #print("----------DecisionTreeUnpruned----------")
    #classifier.set_property("classifier", base4.jobject)
    #classify(preProcessedData,classifier,True,'unprunedJ48.model',splitPerc=70,randomSeed=10)
    #classify(preProcessedData,classifier,False,'unprunedJ48.model',splitPerc=70,randomSeed=10)

    #Hoeffding tree - cross validate - traintestSplit
    #print("----------HoeffdingTree----------")
    #classify(preProcessedData,base5,True,'HoeffdingTree.model',splitPerc=70,randomSeed=10)
    #classify(preProcessedData,base5,False,'HoeffdingTree.model',splitPerc=70,randomSeed=10)

    #K-Nearest-Neighbours - cross validate - traintestSplit
    #print("----------KNN----------")
    #classifier.set_property("classifier", base6.jobject)
    #classify(preProcessedData,classifier,False,'knn.model',splitPerc=70,randomSeed=10)
    #classify(preProcessedData, classifier, True, 'preProcessedJ48.model', splitPerc=70, randomSeed=10)

    # bayesian belief networks - cross validate - traintestSplit
    #print("----------BayesianBelief----------")
    #classifier.set_property("classifier", base7.jobject)
    #classify(preProcessedData, classifier, True, 'bayesianBelief.model', splitPerc=70, randomSeed=10)
    #classify(preProcessedData, classifier, False, 'bayesianBelief.model', splitPerc=70, randomSeed=10)
