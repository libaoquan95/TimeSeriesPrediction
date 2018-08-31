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

    def deNormaliseWindowsV2(self, windows, bases):
        deNormalisedData = []
        if(len(windows) != len(bases)):
            print ("长度不匹配")
            return []
        for i in range(len(windows)):
            window = windows[i]
            base = bases[i]
            deNormalisedWindow = [(float(base) * (float(p) + 1)) for p in window]
            deNormalisedData.append(deNormalisedWindow)
        return deNormalisedData

    def SplitData(self, windows, test_count):
        windowData = np.array(windows)
        row = windowData.shape[0] - test_count
        train = windowData[:int(row), :]
        xTrain = train[:, :-1]
        yTrain = train[:, -1]
        xTest = windowData[int(row):, :-1]
        yTest = windowData[int(row):, -1]
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
        xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))  
        return [xTrain, yTrain, xTest, yTest]

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
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        predicted = model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted