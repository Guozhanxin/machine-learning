# 实现 AND 操作的感知机示例程序

# 感知机的类
class Perceptron(object):
    def __init__(self, input_para_num, acti_func):
        self.activator = acti_func
        # 权重向量初始化为 0
        self.weights = [0.0 for _ in range(input_para_num)]
    def __str__(self):
        return 'final weights\n\tw0 = {:.2f}\n\tw1 = {:0.2f}\n\tw2 = {:.2f}'\
            .format(self.weights[0],self.weights[1],self.weights[2])

    def predict(self, row_vec):
        act_values = 0.0
        for i in range(len(self.weights)):
            act_values += self.weights[i] * row_vec[i]
        return self.activator(act_values)

    def train(self, dataset, iteration, rate):
        for i in range(iteration):
            for input_vec_label in dataset:
                # 计算感知机在当前权重下的输出
                prediction = self.predict(input_vec_label)
                # 更新权重
                self._update_weights(input_vec_label, prediction, rate)

    # “_” 开头的方法是类中的私有方法
    def _update_weights(self, input_vec_label, prediction, rate):
        delta = input_vec_label[-1] - prediction
        for i in range(len(self.weights)):
            self.weights[i] += rate * delta * input_vec_label[i]

def func_activator(input_value):
    return 1.0 if input_value >= 0.0 else 0.0

def get_training_dataset():
    # 构建训练数据 第一项 w0 永远为 -1
    dataset = [[-1, 1, 1, 1], [-1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 1, 0]]
    # 期望的输出列表，注意要与输出一一对应
    # [-1, 1, 1] -> 1,[-1, 0, 0] -> 0, [-1, 1, 0] -> 0, [-1, 0, 1] -> 0.
    return dataset

def train_and_perceptron():
    p = Perceptron(3, func_activator)
    # 获取训练数据
    dataset = get_training_dataset()
    p.train(dataset, 10, 0.1) #指定迭代次数：10轮，学习率设置为0.1
    #返回训练好的感知机
    return p

if __name__ == "__main__":
    # 训练 and 感知机
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([-1, 1, 1]))
    print('0 and 0 = %d' % and_perception.predict([-1, 0, 0]))
    print('1 and 0 = %d' % and_perception.predict([-1, 1, 0]))
    print('0 and 1 = %d' % and_perception.predict([-1, 0, 1]))
