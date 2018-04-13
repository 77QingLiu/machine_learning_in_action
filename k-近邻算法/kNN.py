from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
import os
os.chdir(r'C:\Users\c244032\Documents\GitHub\machine_learning_in_action\k-近邻算法')

class kNN:
    def __init__(self, filename):
        self.filename = filename

    def plotMatrix(self):
        datingMatrix, datingLabels = self._file2matrix(self.filename)
        fig                        = plt.figure()
        ax                         = fig.add_subplot(111)
        ax.scatter(datingMatrix[:, 0], datingMatrix[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
        plt.show()

    def classifyPerson(self):
        resultList                  = ['not at all', 'in small doses', 'in large doses']
        percentTats                 = float(input("percentage of time spent playing video games ?"))
        ffMiles                     = float(input("frequent filer miles earned per year?"))
        iceCream                    = float(input("liters of ice cream consumed per year?"))
        datingDataMat, datingLabels = self._file2matrix(self.filename)
        normMat, ranges, minVals    = self._autoNorm(datingDataMat)
        inArr                       = array([ffMiles, percentTats, iceCream])
        classifierResult            = self._classify0((inArr-minVals)/ranges,normMat,datingLabels, 3)
        print("You will probably like this person: ", resultList[classifierResult - 1])

    def _file2matrix(self, filename):
        """
        Desc:
            导入训练数据
        parameters:
            filename: 数据集路径
        return:
            数据矩阵 returnMatrix 和对应的类别 classLabelVector
        """
        fr = open(filename)
        # 获得文件中的数据行的行数
        numberOfLines = len(fr.readlines())
        # 生成对应的空矩阵
        # 例如: zeros(2，3)就是生成一个 2*3的矩阵，各个位置上全是 0
        returnMatrix     = zeros((numberOfLines, 3))
        classLabelVector = []   # prepare labels return
        index = 0
        fr = open(filename)
        for line in fr.readlines():
            line         = line.strip()
            listFromLine = line.split('\t')
            # 每列的属性数据
            returnMatrix[index, :] = listFromLine[0:3]
            # 每列的类别数据，就是 label 标签数据
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        # 返回数据矩阵returnMat和对应的类别classLabelVector
        return returnMatrix, classLabelVector

    def _autoNorm(self, dataSet):
        """
        Desc:
            归一化特征值，消除特征之间量级不同导致的影响
        parameter:
            dataSet: 数据集
        return:
            归一化后的数据集 normDataSet.  ranges和minVals即最小值与范围.

        归一化公式：
            Y = (X-Xmin)/(Xmax-Xmin)
            其中的 min 和 max 分别是数据集中的最小特征值和最大特征值。该函数可以自动将数字特征值转化为0到1的区间。
        """
        # 计算每种属性的最大值、最小值、范围
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        # 极差
        ranges      = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m           = dataSet.shape[0]
        # 生成与最小值之差组成的矩阵
        normDataSet = dataSet - tile(minVals, (m,1))
        # 将最小值之差除以范围组成矩阵
        normDataSet = normDataSet / tile(ranges, (m,1))
        return normDataSet, ranges, minVals

    def _classify0(self, inX, dataSet, labels, k):
        """
        对于每一个在数据集中的数据点：
            计算目标的数据点（需要分类的数据点）与该数据点的距离
            将距离排序：从小到大
            选取前K个最短距离
            选取这K个中最多的分类类别
            返回该类别来作为目标数据点的预测值
        """
        dataSetSize = dataSet.shape[0]
        # 距离度量 度量公式为欧氏距离
        diffMat     = tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat   = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances   = sqDistances**0.5

        # 将距离排序：从小到大
        sortedDistIndicies = distances.argsort()
        #选取前K个最短距离， 选取这K个中最多的分类类别
        classCount={}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


a = kNN(r'data\datingTestSet2.txt')
a.plotMatrix()
a.classifyPerson()
