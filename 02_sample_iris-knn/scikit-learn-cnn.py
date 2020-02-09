#使用scikit-learn实现的K-近邻算法

import numpy as np
import pandas as pd 
from sklearn import datasets
#加载 iRIS 数据集合
scikit_iris = datasets.load_iris()
#转换为pandas的DataFrame格式，以便于观察数据
pd_iris = pd.DataFrame(
    data=np.c_[scikit_iris['data'], scikit_iris['target']],
    columns=np.append(scikit_iris.feature_names, ['y'])
)
print(pd_iris.head(3))

#选择全部特征值参与训练模型
x=pd_iris[scikit_iris.feature_names]
y=pd_iris['y']

#(1)选择模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
#(2)拟合模型
knn.fit(x,y)
#(3)预测新数据
#knn.predict([[4,3,5,3]])
print(knn.predict([[4,3,5,3]])) #通常将用方括号([])括起来的数据理解为列表。为避免歧义，我们再在外面添加一对方括号[[特征1，特征2，...]]来表示类数组。