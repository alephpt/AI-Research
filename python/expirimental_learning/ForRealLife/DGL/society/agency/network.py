import numpy as np
from DGL.cosmos.closet import sigmoid, sigmoid_derivative, tanh, relu, softmax

from DGL.cosmos import Settings

input_size = Settings.N_INS.value
output_size = Settings.N_OUTS.value
l1_size = Settings.HL1.value
l2_size = Settings.HL2.value

parameters = lambda s1, s2: (np.random.rand(s1, s2), np.random.rand(s2))
'''Returns a tuple of random weights and biases for a layer of a neural network'''

class DRL:
    def __init__(self, activation, derivative):
        self.learning_rate = Settings.ALPHA.value
        self.activation = activation
        self.activation_derivative = derivative
        self.in_to_l1_weights, self.in_to_l1_bias = parameters(input_size, l1_size)
        self.l1_to_l2_weights, self.l1_to_l2_bias = parameters(l1_size, l2_size)
        self.l2_to_out_weights, self.l2_to_out_bias = parameters(l2_size, output_size)

            
    def forward(self, inputs):
        '''
        Returns: hidden_1, hidden_2 and output for evaluation'''
        self.hidden_1 = self.activation(np.dot(inputs, self.in_to_l1_weights) + self.in_to_l1_bias)
        self.hidden_2 = self.activation(np.dot(self.hidden_1, self.l1_to_l2_weights) + self.l1_to_l2_bias)
        self.output = self.activation(np.dot(self.hidden_2, self.l2_to_out_weights) + self.l2_to_out_bias)
        return self.output

    def backprop(self, inputs, target):
        '''
        Backpropagates the error across the network'''
        error = target - self.output
        delta = error * self.activation_derivative(self.output)

        h2_delta = delta.dot(self.l2_to_out_weights.T) * sigmoid_derivative(self.hidden_2)
        h1_delta = h2_delta.dot(self.l1_to_l2_weights.T) * sigmoid_derivative(self.hidden_1)

        self.in_to_l1_weights += inputs.T.dot(h1_delta) * self.learning_rate
        self.in_to_l1_bias += np.sum(h1_delta, axis=0) * self.learning_rate

        self.l1_to_l2_weights += self.hidden_1.T.dot(h2_delta) * self.learning_rate
        self.l1_to_l2_bias += np.sum(h2_delta, axis=0) * self.learning_rate

        self.l2_to_out_weights += self.hidden_2.T.dot(delta) * self.learning_rate
        self.l2_to_out_bias += np.sum(delta, axis=0) * self.learning_rate



