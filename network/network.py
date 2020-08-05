
class Network:
    def __init__(self):
        self.layers = []
        self.cost_func = None
        self.cost_func_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def setup_cost_func(self, cost_func, cost_func_prime):
        self.cost_func = cost_func
        self.cost_func_prime = cost_func_prime

    def predict(self, input):
        n = len(input)
        result = []
        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result

    def train(self, x_train, y_train, learning_rate, epochs):
        n = len(x_train)
        for i in range(epochs):
            sample_error = 0

            for j in range(n):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                sample_error += self.cost_func(y_train[j], output)
                layer_error = self.cost_func_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    layer_error = layer.backward_propagation(layer_error, learning_rate)
            sample_error /= n

            print('epoch :%d/%d error = %f' % (i, epochs, sample_error))
