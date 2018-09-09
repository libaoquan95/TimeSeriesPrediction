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
    testCount    = int(sys.argv[2])

    epochs = config.epochs
    seqLen = config.seqLen
    sensitivity = config.sensitivity

    global_start_time = time.time()
    print('> Loading data... ')
    indiactorTableName = config.basePath+config.indiactorTableName
    indicatorTable = IndicatorTable(indiactorTableName)
    indicator = indicatorTable.getIndicatorByKey(indicatorKey)

    lstm = lstm()
    windows = lstm.transformDataToWindows(lstm.normalisMinAndMax(indicator.df[indicator.valueName].tolist()), seqLen)
    # 预测最后testCount天
    xTrain, yTrain, xTest, yTest = lstm.SplitData(windows, testCount)
 
    print('> Data Loaded. Compiling...')
    model = lstm.buildModel([1, seqLen, 100, 1])
    model.fit(
        xTrain,
        yTrain,
        batch_size=512,
        nb_epoch=epochs,
        validation_split=0.05)
    predicted = lstm.predictPointByPoint(model, xTest)
    testWindowBound = []
    tempXTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1]))
    for i in range(tempXTest.shape[0]):
        min_ = min(tempXTest[i])
        max_ = max(tempXTest[i])
        testWindowBound.append([min_, max_])

    hitScore, hitInfo = lstm.score(yTest, predicted, testWindowBound, sensitivity)
    print (hitInfo)
    print ('hit score: %f' % hitScore)

    print('Training duration (s) : ', time.time() - global_start_time)
    DrawImage(indiactorTableName+"_"+str(indicatorKey), "img/").plotTwoV2(yTest, "True Date", predicted, "Predicted", hitInfo)