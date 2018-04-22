from numpy import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
import operator
import os
os.chdir(r'C:\Users\chase\Documents\GitHub\machine_learning_in_action\k-近邻算法')


class kNN:
    """
    kNN classifier from scratch
    """
    def __init__(self, k):
        self.neighbor = k
        self.data = self._loadData()

    def kNearestNeighbor(self):
        X_train, X_test, Y_train, Y_test = self.data

    	# normalize the matrix
        X_train = self._autoNorm(X_train)
        X_test = self._autoNorm(X_test)

    	# loop over all observations
        predictions = []
        for i in range(len(X_test)):
            predictions.append(self.predict(X_train, Y_train, X_test[i, :], self.neighbor))

        predictions = np.asarray(predictions)
        # evaluating accuracy
        accuracy = accuracy_score(Y_test, predictions)
        print('\nThe accuracy of our classifier is {0}%'.format(accuracy*100))

    def predict(self, X_train, Y_train, X_predict, k):
        """
        对于每一个在数据集中的数据点：
            计算目标的数据点（需要分类的数据点）与该数据点的距离
            将距离排序：从小到大
            选取前K个最短距离
            选取这K个中最多的分类类别
            返回该类别来作为目标数据点的预测值
        """
    	# create list for distances and targets
        distances = []
        targets = []

        # 距离度量 度量公式为欧氏距离
        diffMat     = tile(X_predict, (X_train.shape[0], 1)) - X_train
        sqDiffMat   = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances   = sqDistances**0.5
        # 将距离排序：从小到大
        sortedDistIndicies = distances.argsort()
        #选取前K个最短距离， 选取这K个中最多的分类类别
        classCount={}
        for i in range(k):
            voteIlabel = Y_train[sortedDistIndicies[i]][0]
            classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def plotMatrix(self):
        X_train, X_test, Y_train, Y_test = self.data
        fig                        = plt.figure()
        ax                         = fig.add_subplot(111)
        ax.scatter(X_train[:, 0], X_train[:, 1], 15.0*array(Y_train), 15.0*array(Y_train))
        plt.show()

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
        return normDataSet

    def _loadData(self):
        iris = datasets.load_iris()
        iris.target.shape = (iris.target.shape[0], 1)
        X = iris.data
        Y = iris.target
        # split into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        return X_train, X_test, Y_train, Y_test

a = kNN(3)
a.plotMatrix()

class kNN_scilearn:
    def __init__(self):
        pass

    def fitModel(self):
        X_train, X_test, Y_train, Y_test = self._loadData()
        # MinMax标准化
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.fit_transform(X_test)

        # instantiate learning model (k = 3)
        knn = KNeighborsClassifier(n_neighbors=3)
        # fitting the model
        knn.fit(X_train, Y_train)
        # predict the response
        pred = knn.predict(X_test)
        # evaluate accuracy
        print(accuracy_score(Y_test, pred))
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        # reshape to (n, )
        Y  = Y .reshape(Y.shape[0],)
        self.crossValidation(X, Y)

    def _loadData(self):
        iris = datasets.load_iris()
        iris.target.shape = (iris.target.shape[0], 1)
        X = iris.data
        Y = iris.target
        # split into train and test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        return X_train, X_test, Y_train, Y_test

    def _print(self):
        data = np.append(iris.data, iris.target, axis=1)

    def crossValidation(self, X_train, Y_train):
        # creating odd list of K for KNN
        myList = list(range(1,50))

        # subsetting just the odd ones
        neighbors = list(filter(lambda x: x % 2 != 0, myList))

        # empty list that will hold cv scores
        cv_scores = []

        # perform 10-fold cross validation
        for k in neighbors:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())

        # changing to misclassification error
        MSE = [1 - x for x in cv_scores]

        # determining best k
        optimal_k = neighbors[MSE.index(min(MSE))]
        print("The optimal number of neighbors is {0}".format(optimal_k))

        # plot misclassification error vs k
        plt.plot(neighbors, MSE)
        plt.xlabel('Number of Neighbors K')
        plt.ylabel('Misclassification Error')
        plt.show()

a = kNN_scilearn()
a.fitModel()

a.plotMatrix()
a.classifyPerson()
