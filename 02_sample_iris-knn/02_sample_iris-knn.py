import operator
import csv
import math
import random

# 加载本地文件中的数据
def load_local_data(filename, split, training_set=[], test_set=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(len(dataset[x]) - 1):
                dataset[x][y] = float(dataset[x][y])

            if random.random() < split:          # random() 生成 [0,1] 区间的随机数
                training_set.append(dataset[x])
            else:
                test_set.append(dataset[x])

# 计算欧几里得距离，维度 length
def Euclid_dist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(instance1[x] - instance2[x], 2)
    return math.sqrt(distance)

# 获取邻居
def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = Euclid_dist(test_instance, training_set[x], length)
        distances.append((training_set[x], dist)) # 构成拥有两个成员的元组([],dist)
    distances.sort(key = operator.itemgetter(1))  # 以第1列为关键字排序
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# 根据邻居进行分类
def get_class(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        instance_class = neighbors[x][-1]
        if instance_class in class_votes:
            class_votes[instance_class] += 1
        else:
            class_votes[instance_class] = 1
    # 根据邻居分属进行排序
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0] # 返回排序最靠前的

# 计算正确率
def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if(test_set[x][-1] == predictions[x]):
            correct += 1
    return (correct/float(len(test_set))) * 100.0

def main():
    training_set=[]
    test_set = []
    split = 0.7
    load_local_data('iris.csv', split, training_set, test_set)
    print('训练集合：' + repr(len(training_set)))
    print('测试集合：' + repr(len(test_set)))
    predictions = []
    k = 3
    for x in range(len(test_set)):
        neighbors = get_neighbors(training_set, test_set[x], k)
        result = get_class(neighbors)
        predictions.append(result)
        print('> 预测=' + repr(result) + ', 实际=' + repr(test_set[x][-1]))
    accuracy = get_accuracy(test_set, predictions)
    print('精确度为：' + repr(accuracy) + '%')

main()
