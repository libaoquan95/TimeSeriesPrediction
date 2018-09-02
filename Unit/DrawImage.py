import matplotlib.pyplot as plt

class DrawImage():
    def __init__(self, title, saveBasePath):
        self.title = title
        self.saveBasePath = saveBasePath
    def plotOne(self, y, label=""):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        plt.plot(y, '-o', label=label)
        plt.title(self.title)
        plt.legend()
        plt.show()
    def plotTwo(self, y1, label1, y2, label2):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        plt.plot(y1, '-o', label=label1)
        plt.plot(y2, '-o', label=label2)
        plt.title(self.title)
        plt.legend()
        plt.show()