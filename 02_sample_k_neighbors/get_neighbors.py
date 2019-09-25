import operator
import math

def Euclid_dist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow(instance1[x] - instance2[x], 2) # &(0,n)(x[i] - y[i])^2
    return math.sqrt(distance)

def get_neighbors(training_set, test_instance, k):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(training_set)):
        dist = Euclid_dist(test_instance, training_set[x], length)
        distances.append((training_set[x], dist)) # 构成拥有两个成员的元组([],dist)
        print(distances)
        print('...')
    distances.sort(key = operator.itemgetter(1))  # 以第1列为关键字排序
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors

training_set = [[3, 2, 6, 'a'], [1, 2, 4, 'b'], [2, 2, 2, 'b'], [1, 5, 4, 'a']]
test_instance = [4, 6, 7]
k = 1
neighbors = get_neighbors(training_set, test_instance, k)
print('测试样本最近的邻居为：', neighbors)
