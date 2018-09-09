import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

class lstm():
    def transformDataToWindows(self, data, seq_len):
        for i in range(len(data)):
            try:
                data[i] = float(data[i])
            except ValueError:
                print (data[i])
        sequence_length = seq_len + 1
        windows = []
        for index in range(len(data) - seq_len):
            windows.append(data[index: index + sequence_length])
        return windows

    def normalisMinAndMax(self, data):
        max_ = min_ = data[0]
        for d in data:
            max_ = max(max_, d)
            min_ = min(min_, d)
        normalisedData = []
        for d in data:
            normalisedData.append(float(d-min_+1)/(max_-min_+1))
        return normalisedData
    '''
    def normaliseWindows(self, windows):
        normalisedData = []
        normalisedBase = []
        for window in windows:
            mean = self.getMean(window)
            normalisedBase.append(mean)
            normalisedWindow = [((float(p) / float(mean)) - 1) for p in window]
            normalisedData.append(normalisedWindow)
        return [normalisedData, normalisedBase]

    def getMean(self, datas):
        maxData = datas[0]
        minData = datas[0]
        for data in datas:
            maxData = max(data, maxData)
            minData = min(data, minData)
        return (maxData+minData)/2

    def deNormaliseWindows(self, windows, bases):
        deNormalisedData = []
        if(len(windows) != len(bases)):
            print ("长度不匹配")
            return []
        for i in range(len(windows)):
            window = windows[i]
            base = bases[i]
            deNormalisedData.append(float(base) * (float(window) + 1))
        return deNormalisedData
    '''

    def SplitData(self, windows, test_count):
        windowData = np.array(windows)
        row = windowData.shape[0] - test_count
        train = windowData[:int(row), :]
        np.random.shuffle(train)
        xTrain = train[:, :-1]
        yTrain = train[:, -1]
        xTest = windowData[int(row):, :-1]
        yTest = windowData[int(row):, -1]
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
        xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))  
        return [xTrain, yTrain, xTest, yTest]

    def SplitDataV2(self, windows):
        windowData = np.array(windows)
        X = windowData[:, :-1]
        Y = windowData[:, -1]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1)) 
        return [X, Y]

    def buildModel(self, layers):
        model = Sequential()

        model.add(LSTM(
            input_shape=(layers[1], layers[0]),
            output_dim=layers[1],
            return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(
            layers[2],
            return_sequences=False))
        model.add(Dropout(0.2))

        model.add(Dense(
            output_dim=layers[3]))
        model.add(Activation("linear"))

        start = time.time()
        model.compile(loss="mse", optimizer="rmsprop")
        print("> Compilation Time : ", time.time() - start)
        return model

    def predictPointByPoint(self, model, data):
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def score(self, real, pred, bound, sensitivity=0.1):
        hitCount = 0
        isHit = []
        if len(real) != len(pred) or len(real) != len(bound):
            print ("长度不匹配")
        for i in range(len(real)):
            if (abs(bound[i][0] - bound[i][1] == 0)):
                if bound[i][0] == real[i]:
                    hitCount = hitCount + 1
                    isHit.append(1)
                else: 
                    isHit.append(0)
            else:
                if (abs(real[i] - pred[i]) <= sensitivity * abs(bound[i][0] - bound[i][1])):
                    hitCount = hitCount + 1
                    isHit.append(1)
                else:
                    isHit.append(0)
        return [float(hitCount)/len(real), isHit]