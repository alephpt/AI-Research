from .utilities import *
from .logger import Log

import numpy as n

def tanh(x):
    return n.tanh(x)

def sigmoid(x):
    return 1 / (1 + n.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return n.maximum(0, x)

def softmax(x):
    exps = n.exp(x - n.max(x))
    return exps / n.sum(exps)

# TODO: Audio/Signals w/ Nova Engine
def cross_entropy_loss(y, y_hat):
    return -n.sum(y * n.log(y_hat))