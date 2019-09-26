import csv
import random

def load_local_data(filename, split, training_set=[], test_set=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])

            if random.random() < split:          # random() 生成 [0,1] 区间的随机数
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])

training_set = []
testset = []
load_local_data('iris.csv', 0.70, training_set, testset) # 列表是传引用
print('训练集合样本数：' + repr(len(training_set)))        # repr() 转换为供解释器读取的类型
print('测试集合样本数：' + repr(len(testset)))
