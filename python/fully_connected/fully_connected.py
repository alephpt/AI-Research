import numpy as n

def tanh(x):
    return n.tanh(x)

def tanh_prime(x):
    return 1 - n.tanh(x) ** 2

def mse(y_returned, y_expected):
    return n.mean(n.power(y_returned - y_expected, 2))

def mse_prime(y_returned, y_expected):
    return 2 * (y_expected - y_returned) / y_returned.size


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, data_in):
        raise NotImplementedError
    
    def backward(self, err_out, learning_rate):
        raise NotImplementedError
    
    
class FC(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.weights = n.random.rand(n_inputs, n_outputs) - 0.5
        self.bias = n.random.rand(1, n_outputs) - 0.5
        
    def forward(self, data_in):
        self.input = data_in
        self.output = n.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backward(self, err_out, learning_rate):
        self.weights -= learning_rate * n.dot(self.input.T, err_out)
        self.bias -= learning_rate * err_out
        return n.dot(err_out, self.weights.T)
    
    
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.prime = activation_prime
    
    def forward(self, data_in):
        self.input = data_in
        self.output = self.activation(input)
        return self.output
    
    def backward(self, err_out, learning_rate):
        return self.prime(self.input) * err_out