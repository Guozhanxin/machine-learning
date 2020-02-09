#基于scikit-learn的模型评估

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
X=pd_iris[scikit_iris.feature_names]
y=pd_iris['y']

#from sklearn.cross_validation import train_test_split #scikit 0.16.1
from sklearn.model_selection import train_test_split #scikit 0.22.1
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)#X:样本特征集，y:样本结果，样本占比，随机数种子

#(1)选择模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
#(2)拟合模型
knn.fit(X_train,y_train)
#(3)预测新数据
knn.predict([[4,3,5,3]])
y_predict_on_train = knn.predict(X_train)
y_predict_on_test = knn.predict(X_test)

print('准确率为：{}'.format(metrics.accuracy_score(y_train,y_predict_on_train)))
print('准确率为：{}'.format(metrics.accuracy_score(y_test,y_predict_on_test)))
