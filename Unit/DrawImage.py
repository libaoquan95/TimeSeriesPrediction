import matplotlib.pyplot as plt

class DrawImage():
    def __init__(self, title, saveBasePath):
        self.title = title
        self.saveBasePath = saveBasePath
    def plotOne(self, y, label=""):
        fig = plt.figure(facecolor='white', figsize=(18,6))
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
    def plotTwoV2(self, y1, label1, y2, label2, y3, label3="", label4=""):
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        plt.plot(y1, '-o', label=label1)
        plt.plot(y2, '-', label=label2)
        for i in range(len(y3)):
            if (y3[i] == 0):
                plt.scatter(i, y2[i], color='k')
            else:
                plt.scatter(i, y2[i], color='orange')
        plt.title(self.title)
        plt.legend()
        plt.show()

def main():
    a = []
    b = []
    for i in range(5):
        a.append(i)
        b.append(i*2)
    c = [1,0,1,0,1]
    di = DrawImage("test", "test")
    di.plotTwoV2(a, "a", b, "b", c, "not hit", "hit")

#main()