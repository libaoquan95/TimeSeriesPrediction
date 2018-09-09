import sys
import time
import numpy as np
from Model.LSTM.lstm import lstm
from Unit.DrawImage import DrawImage
from Unit.Indicator import Indicator
from Unit.Indicator import IndicatorTable
import config

if __name__ == '__main__':
    indicatorKey = int(sys.argv[1])
    baseTrainCount = int(sys.argv[2])
    
    epochs = config.epochs
    seqLen = config.seqLen
    sensitivity = config.sensitivity

    global_start_time = time.time()
    print('> Loading data... ')
    indiactorTableName = config.basePath+config.indiactorTableName
    indicatorTable = IndicatorTable(indiactorTableName)
    indicator = indicatorTable.getIndicatorByKey(indicatorKey)
    lstm = lstm()
    indiactorData = lstm.normalisMinAndMax(indicator.df[indicator.valueName].tolist())
    
    print('> Data size = %d, bengin compiling...' %len(indiactorData))
    testReal = []
    testPred = []
    testWindowBound = []
    for i in range(baseTrainCount, len(indiactorData)):
        windows = lstm.transformDataToWindows(indiactorData[:i], seqLen)
        # 只用一个窗口，预测之后一天
        xTrain, yTrain, xTest, yTest = lstm.SplitData(windows, 1)
        print('> Compiling, data size =  %d...' % i)
        model = lstm.buildModel([1, seqLen, 100, 1])
        model.fit(
            xTrain,
            yTrain,
            batch_size=512,
            nb_epoch=epochs,
            validation_split=0.05)
        predicted = lstm.predictPointByPoint(model, xTest)

        testPred.append(predicted[0])
        testReal.append(yTest[0])
        tempXTest = np.reshape(xTest, (xTest.shape[1]))
        testWindowBound.append([min(tempXTest), max(tempXTest)])

    hitScore, hitInfo = lstm.score(testReal, testPred, testWindowBound, sensitivity)
    print ('score = %f' % hitScore)
    DrawImage(indiactorTableName+"_"+str(indicatorKey), "img/").plotTwoV2(testReal, "True Date", testPred, "Predicted", hitInfo)
    