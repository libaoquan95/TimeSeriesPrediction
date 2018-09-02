import sys
import time
from Model.LSTM.lstm import lstm
from Unit.DrawImage import DrawImage
from Unit.Indicator import Indicator
from Unit.Indicator import IndicatorTable

def fun():
    None

if __name__ == '__main__':
    indicatorKey = int(sys.argv[1])

    basePath = "Data/mart_waimai_crm/"
    indiactorTableName = "indicator_org_id_day_xxsfjye_958_table_prod"
    epochs  = 10
    seq_len = 15
    baseTrainCount = 300
    Sensitivity = 0.1

    global_start_time = time.time()
    print('> Loading data... ')
    indicatorTable = IndicatorTable(basePath+indiactorTableName)
    indicator = indicatorTable.getIndicatorByKey(indicatorKey)
    lstm = lstm()
    windows = lstm.transformDataToWindows(indicator.df[indicator.valueName].tolist(), seq_len)
    normaliseWindows, normaliseBase = lstm.normaliseWindows(windows)

    X, Y = lstm.SplitDataV2(normaliseWindows)
    print ('windoes size: %d' % len(X))
    for i in range(len(X)-baseTrainCount):
        trainCount = baseTrainCount + i
        print ('> Data Loaded. Compiling(count=%d)...' % trainCount)
        model = lstm.buildModel([1, seq_len, 100, 1])
        model.fit(
            X[:trainCount],
            Y[:trainCount],
            batch_size=512,
            nb_epoch=epochs,
            validation_split=0.05)
        #predicted = lstm.predictPointByPoint(model, X[trainCount:])     

    '''
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
    yTest = lstm.deNormaliseWindows(yTest, normaliseBase[len(yTrain):])
    predicted = lstm.deNormaliseWindows(predicted, normaliseBase[len(yTrain):])
    '''