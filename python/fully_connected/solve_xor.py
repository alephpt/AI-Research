import pygame
import numpy as n
import fully_connected as nn

data_in = n.array([[[0,0]], [[0, 1]], [[1, 0]], [[1, 1]]])
data_out = n.array([[[0]], [[1]], [[1]], [[0]]])

test_data = n.array([[[1, 0]], [[0, 0]], [[1, 1]], [[0, 1]], [[1, 0]], [[0, 0]]])

fcnn = nn.Network()

fcnn.addLayer(nn.FC(2, 3))
fcnn.addLayer(nn.Activation(nn.tanh, nn.tanh_prime))
fcnn.addLayer(nn.FC(3, 1))
fcnn.addLayer(nn.Activation(nn.tanh, nn.tanh_prime))

fcnn.setLossFn(nn.mse, nn.mse_prime)
fcnn.train(data_in, data_out, 5000, 0.05)

res = fcnn.analyze(data)