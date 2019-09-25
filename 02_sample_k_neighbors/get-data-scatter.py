#coding=utf-8

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
X = df.iloc[0:150,[0,2]].values

plt.scatter(X[0:50,0], X[:50,1], color = 'blue', marker='x', label='setosa')           
plt.scatter(X[50:100,0], X[50:100,1], color = 'red', marker='o', label='versicolor')   
plt.scatter(X[100:150,0], X[100:150,1], color = 'green', marker='*', label='virginica')

plt.xlabel('petal width')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
plt.savefig('scatter.png')
plt.show()
