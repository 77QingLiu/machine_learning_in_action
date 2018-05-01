import numpy as np
from io import StringIO
import pandas as pd
import math

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

# sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
# re-encoding
columns = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
for col in columns:
    df[col] = LabelEncoder().fit_transform(df[col])
enc = OneHotEncoder().fit(df[columns])
a = enc.transform(df[columns]).toarray()
pd.DataFrame(a)
df
pd.get_dummies(df, columns=columns)


df.head()
青绿 |蜷缩 |浊响 |清晰 |凹陷 |硬滑 |0.697 |0.46
LR = LogisticRegression()
y_pred = LR.fit(df[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']], df['好瓜'])
print('Number of mislabeled points out of a total {0} points : {1}'.format(len(y_pred), sum(y_pred != df['好瓜'])))












data = [ ['cat', 1], ['cat', 2], ['dog', 3], ['cat', 2], ['fish', 0] ]

ndata = np.array(data)
ndata
# 將 cat, dog, fish 分類, 成 0, 1, 2
ndata[:, 0] = LabelEncoder().fit_transform(ndata[:, 0])
ndata
"""
        type
array([['0', '1'],
       ['0', '2'],
       ['1', '3'],
       ['0', '2'],
       ['2', '3']],
      dtype='<U4')
"""
enc = OneHotEncoder().fit(ndata)
enc.n_values_
# array([3, 4])
"""
表示 ndata[0] 有 3 種特徵值
ndata[1] 有 4 種特徵值，分別為 0,1,2,3
"""
enc.feature_indices_
# array([0, 3, 7])
"""
表示特徵的範圍，例如
0-3 為 ndata[0],
3-7 為 ndata[1]
"""
# 將 ndata 放入做測試
print(enc.transform(ndata).toarray())
"""
        cat | dog |fish| 0 | 1 | 2 |  3
array([[ 1.,  0.,  0.,  0.,  1.,  0.,  0.],
       [ 1.,  0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  1.,  0.,  0.,  0.,  0.,  1.],
       [ 1.,  0.,  0.,  0.,  0.,  1.,  0.],
       [ 0.,  0.,  1.,  1.,  0.,  0.,  0.]])
"""
