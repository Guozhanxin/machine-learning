import pandas as pd
import csv

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
print(df.tail())
df.to_csv('iris.csv', header = 0)
