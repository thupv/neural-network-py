from network.network import Network
from layers.FCLayer import FCLayer
from layers.ALayer import ALayer
import numpy as np


def relu_func(z):
    z[z < 0] = 0
    return z


def relu_func_prime(z):
    z[z < 0] = 0
    z[z > 0] = 1
    return z


def cost_func(y_true, y_pred):
    return 0.5 * (y_pred - y_true) ** 2


def cost_func_prime(y_true, y_pred):
    return y_pred - y_true


x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])


network = Network()
network.add(FCLayer((1, 2), (1, 3)))
network.add(ALayer((1, 3), (1, 3), relu_func, relu_func_prime))
network.add(FCLayer((1, 3), (1, 1)))
network.add(ALayer((1, 1), (1, 1), relu_func, relu_func_prime))

network.setup_cost_func(cost_func, cost_func_prime)
network.train(x_train, y_train, epochs=1000, learning_rate=0.01)

out = network.predict([[0, 1]])

print(out)