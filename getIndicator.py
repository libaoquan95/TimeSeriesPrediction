import sys
from Model.LSTM.lstm import lstm
from Unit.DrawImage import DrawImage
from Unit.Indicator import Indicator
from Unit.Indicator import IndicatorTable
import config

if __name__ == '__main__':
    indicatorKey = int(sys.argv[1])

    indiactorTableName = config.basePath+config.indiactorTableName
    indicatorTable = IndicatorTable(indiactorTableName)
    indicator = indicatorTable.getIndicatorByKey(indicatorKey)
    indiactorData = indicator.df[indicator.valueName].tolist()
    #lstm = lstm()
    #indiactorData = lstm.normalisMinAndMax(indiactorData)

    print ("size = %d" % len(indiactorData))
    DrawImage(indiactorTableName+"_"+str(indicatorKey), "img/").plotOne(indiactorData, "True Date")