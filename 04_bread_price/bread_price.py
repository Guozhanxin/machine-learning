# 基于梯度下降的线性回归
# 线性回归方程： y = w1 * x + w0 * 1
# 用线性代数中的矩阵表述为： y = [w1, w0] * [x, 1]T

# 目标：使用梯度下降的方法，根据样本数据，反复迭代获取最佳的 w0,w1。最后得到目标方程。

# 数据
bread_price = [[0.5,5],[0.6,5.5],[0.8,6],[1.1,6.8],[1.4,7]]

# 更新一次 w0, w1 的值
def BGD_step_gradient(w0_current, w1_current, points, learninggRate):
    w0_gradient = 0
    w1_gradient = 0

    # 遍历所有样本数据，计算 grad(w0), grad(w1)
    # grad(Wi) = -1 * sum((target(d) - output(d)) * Xi(d))    Xi(d): Wi 对应的系数，如 w1 对应 x, w0 对应 1.
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]

        #计算当前的梯度
        w0_gradient += -1.0 * (y - ((w1_current * x) + w0_current))
        w1_gradient += -1.0 * x * (y - ((w1_current * x) + w0_current))
    # Wi <-- Wi + n * sum((target(d) - output(d)) * Xi(d))     n: learninggRate
    new_w0 = w0_current - (learninggRate * w0_gradient)
    new_w1 = w1_current - (learninggRate * w1_gradient)
    return [new_w0, new_w1]

# 梯度下降算法
def gradient_descent_runner(points, start_w0, start_w1, l_rate, num_iterations):
    w0 = start_w0
    w1 = start_w1
    for i in range(num_iterations):
        w0, w1 = BGD_step_gradient(w0, w1, points, l_rate)
    return [w0, w1]

def predict(w0, w1, wheat):
    price = w1 * wheat + w0
    return price

if __name__ == "__main__":
    learning_rate = 0.01 # 学习率
    num_iter = 100       # 迭代次数
    w0, w1 = gradient_descent_runner(bread_price, 1, 1, learning_rate, num_iter)
    price = predict(w0, w1, 0.9)  # 预测 0.9 磅面包的价格。
    print("price = ", price)
