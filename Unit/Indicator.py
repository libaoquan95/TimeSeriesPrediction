import pandas as pd

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