#coding=utf-8

from math import sqrt
import matplotlib.pyplot as plt

dataset = [[1.2, 1.1], [2.4, 3.5], [4.1, 3.2], [3.4, 2.8], [5, 5.4]] #样本数据

# 计算样本均值
def mean(values):
    return sum(values) / float(len(values))

# 计算 x 与 y 的协方差
def covariance(x, mean_x, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (x[i] - mean_y)
    return covar

# 计算方差
def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])

# 计算回归系数
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    w1 = covariance(x, x_mean, y_mean) / variance(x, x_mean)
    w0 = y_mean - w1 * x_mean
    return (w0, w1)

#计算均方根误差 RMSE
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += prediction_error
        # print(prediction_error, sum_error)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)

#构建简单线程回归
def simple_linear_regression(train, test):
    predictions = list()
    w0, w1 = coefficients(train)
    for row in test:
        y_model = w1 * row[0] + w0
        predictions.append(y_model)
    return predictions

def show(dataset, predicted):
    plt.axis([0, 6, 0, 6])
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    plt.plot(x, y, 'bs')
    plt.plot(x, predicted, 'ro-')
    plt.grid()
    plt.savefig('scatter.png')
    plt.show()

#评估算法数据准备及协调
def evaluate_algorithm(dataset, algorithm):
    test_set = list()
    for row in dataset:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(dataset, test_set)
    show(dataset, predicted)
    for val in predicted:
        print('%.3f\t'%(val))
    actual = [row[-1] for row in dataset]
    rmse = rmse_metric(actual, predicted)
    return rmse

#返回 RMSE
rmse = evaluate_algorithm(dataset, simple_linear_regression)
print('rmse:%.3f' % (rmse))

