from abc import ABC

from .layer import Layer
import numpy as np


# Activation layer
class ALayer(Layer, ABC):
    def __init__(self, input_shape, output_shape, activation_func, activation_func_prime):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation_func = activation_func
        self.activation_func_prime = activation_func_prime

    def forward_propagation(self, input):
        self.input = input
        self.output = self.activation_func(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_func_prime(self.input) * output_error

