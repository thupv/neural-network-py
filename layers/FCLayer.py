from abc import ABC

from .layer import Layer
import numpy as np


# Full connected layer
class FCLayer(Layer, ABC):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5

    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        current_layer_error = np.dot(output_error, self.weights.T)
        d_weight = np.dot(self.input.T, output_error)

        self.weights -= d_weight*learning_rate
        self.bias -= learning_rate*output_error

        return current_layer_error

