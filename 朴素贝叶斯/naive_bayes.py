import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from io import StringIO
import pandas as pd
import subprocess


def createDataSet():
    ''' 数据读入 '''
    rawData = StringIO(
    """编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜
      1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是
      2,乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,0.774,0.376,是
      3,乌黑,蜷缩,浊响,清晰,凹陷,硬滑,0.634,0.264,是
      4,青绿,蜷缩,沉闷,清晰,凹陷,硬滑,0.608,0.318,是
      5,浅白,蜷缩,浊响,清晰,凹陷,硬滑,0.556,0.215,是
      6,青绿,稍蜷,浊响,清晰,稍凹,软粘,0.403,0.237,是
      7,乌黑,稍蜷,浊响,稍糊,稍凹,软粘,0.481,0.149,是
      8,乌黑,稍蜷,浊响,清晰,稍凹,硬滑,0.437,0.211,是
      9,乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,0.666,0.091,否
      10,青绿,硬挺,清脆,清晰,平坦,软粘,0.243,0.267,否
      11,浅白,硬挺,清脆,模糊,平坦,硬滑,0.245,0.057,否
      12,浅白,蜷缩,浊响,模糊,平坦,软粘,0.343,0.099,否
      13,青绿,稍蜷,浊响,稍糊,凹陷,硬滑,0.639,0.161,否
      14,浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,0.657,0.198,否
      15,乌黑,稍蜷,浊响,清晰,稍凹,软粘,0.36,0.37,否
      16,浅白,蜷缩,浊响,模糊,平坦,硬滑,0.593,0.042,否
      17,青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,0.719,0.103,否
    """)

    df = pd.read_csv(rawData, sep=",")
    return df

df = createDataSet()

df.head()


def naive_bayes(data=, feature=):
    # 估计类先验证概率
    N = len(data)
    #P_pre = data['好瓜'].value_counts().apply(lambda x:'{0:.3f}'.format(x/N))
    for key, val in feature.items():
        data[data[key] == val].groupby('好瓜').

from collections import OrderedDict
pd.Series(OrderedDict({'色泽':'青绿', '根蒂':'蜷缩', '敲声':'浊响', '纹理':'清晰', '脐部':'凹陷', '触感':'硬滑'}))
df.groupby('好瓜')[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']].apply(lambda x:x)
'色泽', '根蒂', '敲声', '纹理', '脐部', '触感'
