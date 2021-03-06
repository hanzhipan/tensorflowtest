# coding=utf-8
import numpy


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + numpy.exp(-inX))


def gradAscend(dataMatIn, classLabels):
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    m, n = numpy.shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weight = numpy.ones((n, 1))
    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weight)
        error = labelMat - h
        weight += alpha * dataMatrix.transpose() * error
        # plotBestFit(weight)
    return weight


def gradAscendWithDraw(dataMatIn, classLabels):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(311, ylabel='x0')
    bx = fig.add_subplot(312, ylabel='x1')
    cx = fig.add_subplot(313, ylabel='x2')
    dataMatrix = numpy.mat(dataMatIn)
    labelMat = numpy.mat(classLabels).transpose()
    m, n = numpy.shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weight = numpy.ones((n, 1))
    wei1 = []
    wei2 = []
    wei3 = []
    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weight)
        error = labelMat - h
        weight += alpha * dataMatrix.transpose() * error
        wei1.extend(weight[0])
        wei2.extend(weight[1])
        wei3.extend(weight[2])
    ax.plot(range(maxCycle), wei1)
    bx.plot(range(maxCycle), wei2)
    cx.plot(range(maxCycle), wei3)
    plt.xlabel('iter_num')
    plt.show()
    return weight


def stocGradAscent0(dataMatrix, classLabels):
    m, n = numpy.shape(dataMatrix)

    alpha = 0.001
    weight = numpy.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weight))
        error = classLabels[i] - h
        weight = weight + alpha * error * dataMatrix[i]
    return weight


def stocGradAscentWithDraw0(dataMatrix, classLabels):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(311, ylabel='x0')
    bx = fig.add_subplot(312, ylabel='x1')
    cx = fig.add_subplot(313, ylabel='x2')
    m, n = numpy.shape(dataMatrix)

    alpha = 0.001
    weight = numpy.ones(n)
    wei1 = numpy.array([])
    wei2 = numpy.array([])
    wei3 = numpy.array([])
    numIter = 200
    for j in range(numIter):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i] * weight))
            error = classLabels[i] - h
            weight = weight + alpha * error * dataMatrix[i]
            wei1 = numpy.append(wei1, weight[0])
            wei2 = numpy.append(wei2, weight[1])
            wei3 = numpy.append(wei3, weight[2])
    ax.plot(numpy.array(range(m * numIter)), wei1)
    bx.plot(numpy.array(range(m * numIter)), wei2)
    cx.plot(numpy.array(range(m * numIter)), wei3)
    plt.xlabel('iter_num')
    plt.show()
    return weight


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = numpy.shape(dataMatrix)

    # alpha = 0.001
    weight = numpy.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(numpy.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weight))
            error = classLabels[randIndex] - h
            weight = weight + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weight


def stocGradAscentWithDraw1(dataMatrix, classLabels, numIter=150):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(311, ylabel='x0')
    bx = fig.add_subplot(312, ylabel='x1')
    cx = fig.add_subplot(313, ylabel='x2')
    m, n = numpy.shape(dataMatrix)

    # alpha = 0.001
    weight = numpy.ones(n)
    wei1 = numpy.array([])
    wei2 = numpy.array([])
    wei3 = numpy.array([])
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(numpy.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weight))
            error = classLabels[randIndex] - h
            weight = weight + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
            wei1 = numpy.append(wei1, weight[0])
            wei2 = numpy.append(wei2, weight[1])
            wei3 = numpy.append(wei3, weight[2])
    ax.plot(numpy.array(range(len(wei1))), wei1)
    bx.plot(numpy.array(range(len(wei2))), wei2)
    cx.plot(numpy.array(range(len(wei2))), wei3)
    plt.xlabel('iter_num')
    plt.show()
    return weight


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weight = wei
    dataMat, labelMat = loadDataSet()
    dataArr = numpy.array(dataMat)
    n = numpy.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = numpy.arange(-3.0, 3.0, 0.1)
    y = (-weight[0] - weight[1] * x) / weight[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def main():
    dataArr, labelMat = loadDataSet()
    # w = gradAscendWithDraw(dataArr,labelMat)
    w = stocGradAscentWithDraw0(numpy.array(dataArr), labelMat)
    plotBestFit(w)


if __name__ == '__main__':
    main()
