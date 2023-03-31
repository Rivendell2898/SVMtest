import numpy as np
from matplotlib import pyplot as plt
import cvxopt

def reg_func(x_data, y_data, alpha, b, x, n=1):
    '''
    使用SVM进行回归拟合的函数
    :param x_data: 数据集x
    :param y_data: 数据集y
    :param alpha: 计算出的alpha向量
    :param b: 计算出的偏置项
    :param x: 需要拟合的变量
    :param n: 核函数阶数
    :return: 拟合后的函数值
    '''
    N = x_data.shape[0]  # 从输入数据中获得数据集大小
    tmp = 0.0
    for i in range(N):
        tmp += alpha[i] * y_data[i] * kernel1(x_data[i], x, n)

    return tmp + b


def kernel1(x, y, n=1):
    '''
    核函数(1+x*y)^n
    :param x: 核函数变量x
    :param y: 核函数变量y
    :param n: 阶数
    :return: 和函数值
    '''
    return (1 + x*y)**n

# 生成数据集，设置参数
# 使用随机种子使每次生成的数据都相同
N = 30
lam = 1.0
np.random.seed(1)
x = np.random.uniform(low=0, high=1, size=N)
np.random.seed(1)
epsilon = np.random.normal(size=N)
y = -5 + 20*x - 16*x*x + 0.4*epsilon
plt.scatter(x, y)

xx = np.linspace(0, 1, 100)
plt.plot(xx, (-5 + 20*xx - 16*xx*xx), color="k", label="original function")

# 比较不同lambda下的拟合曲线形态
for n in [1, 2, 9]:
    C = 1 / lam
    # 根据本次作业中word的公式计算SVM的系数矩阵
    Y = np.diag(y)
    # G = np.dot(x.reshape(-1, 1), x.reshape(1, -1))
    G = np.zeros([N, N], dype="float32")
    for i in range(N):
        for j in range(N):
            G[i][j] = kernel1(x[i], x[j], n)

    P = Y @ G @ Y
    q = np.ones(N) * -1
    I = np.identity(N)
    G1 = np.concatenate((I, -I))
    h = np.concatenate((C * np.ones(N), np.zeros(N)))
    A = y.reshape(1, -1)
    b1 = np.zeros(1)

    # 使用qp求解器求解参数alpha
    cvxopt.solvers.options['show_progress'] = False  # 隐藏求解进度信息
    P = cvxopt.matrix(P)
    q = cvxopt.matrix(q)
    G1 = cvxopt.matrix(G1)
    h = cvxopt.matrix(h)
    A = cvxopt.matrix(A)
    b1 = cvxopt.matrix(b1)
    solution = cvxopt.solvers.qp(P, q, G1, h, A, b1)

    # 计算参数alpha和b
    alpha = np.array(solution['x']).reshape(-1)

    bk = np.zeros(N)

    for j in range(N):
        w = 0.0
        for i in range(N):
            w += alpha[i] * y[i] * kernel1(x[i], x[j], n)
        bk[j] = y[j] - w * x[j]

    b = bk.mean()

    # 根据计算出的最终参数绘制拟合曲线


    plt.plot(xx, reg_func(x, y, alpha, b, xx, n), label="n={}".format(n))


plt.xlabel("x value")
plt.ylabel("y value")
plt.legend()
plt.show()
