import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sys
import matplotlib.dates as mdates
from datetime import datetime
from sklearn.pipeline import Pipeline
from LSTM import lstm as lstm
import time

class Indicator:
    def __init__(self, keyName, valueName, dateName, df):
        self.keyName = keyName
        self.valueName = valueName
        self.dateName = dateName
        self.df = df
    def fliterByData(self, beginData, endData):
        df = self.df[(self.df[self.dateName]>=beginDate) & (self.df[self.dateName]<endDate)]
        return Indicator(self.keyName, self.valueName, self.dateName, df)
    
class IndicatorTable():
    def __init__(self, indiactorTableName, delimiter="\t", header=0):
        df = pd.read_csv(indiactorTableName, delimiter=delimiter, header=header)
        headerName = df.columns.values.tolist()
        self.keyName   = headerName[0]
        self.valueName = headerName[1]
        self.dateName  = headerName[2]
        #self.df = df.fillna(0) # NULL值补0
        self.df = df[pd.isnull(df[self.valueName])==False] #过滤NULL
    def getIndicatorByKey(self, key):
        df = self.df[self.df[self.keyName] == key]
        return Indicator(self.keyName, self.valueName, self.dateName, df)
    def getKeys(self):
        return self.df[self.keyName].drop_duplicates()

class DrawImage():
    def __init__(self, title, saveBasePath):
        self.title = title
        self.saveBasePath = saveBasePath
    def plotOne(self, y, label=""):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        plt.plot(y, label=label)
        plt.title(self.title)
        plt.legend()
        plt.show()
    def plotTwo(self, y1, label1, y2, label2):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        plt.plot(y1, label=label1)
        plt.plot(y2, label=label2)
        plt.title(self.title)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    indicatorKey = int(sys.argv[1])
    testCount    = int(sys.argv[2])

    basePath = "mart_waimai_crm/"
    indiactorTableName = "indicator_org_id_day_xxsfjye_958_table_prod"
    epochs  = 100
    seq_len = 15

    global_start_time = time.time()
    print('> Loading data... ')
    indicatorTable = IndicatorTable(basePath+indiactorTableName)
    indicator = indicatorTable.getIndicatorByKey(indicatorKey)
    lstm = lstm.lstm()
    windows = lstm.transformDataToWindows(indicator.df[indicator.valueName].tolist(), seq_len)
    normaliseWindows, normaliseBase = lstm.normaliseWindows(windows)
    xTrain, yTrain, xTest, yTest = lstm.SplitData(normaliseWindows, testCount)
    
    print('> Data Loaded. Compiling...')
    model = lstm.buildModel([1, seq_len, 100, 1])
    model.fit(
        xTrain,
        yTrain,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)
    predicted = lstm.predictPointByPoint(model, xTest)        

    print('Training duration (s) : ', time.time() - global_start_time)
    #yTest = lstm.deNormaliseWindows(yTest, normaliseBase[len(yTrain):])
    #predicted = lstm.deNormaliseWindows(predicted, normaliseBase[len(yTrain):])

    DrawImage(indiactorTableName+"_"+str(indicatorKey), "img/").plotTwo(yTest, "True Date", predicted, "Predicted")
