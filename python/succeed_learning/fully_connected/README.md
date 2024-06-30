# Fully Connected Feed Forward XOR Solving

## Description
These Python scripts demonstrate a simple feedforward neural network (fully connected) implementation using numpy. 

### Dependencies
- Python 3.x
- numpy
- Pygame (for visualization, assumed but not implemented in provided script)

### Execution
`python solve_xor.py`

---

# fully_connected.py
This Python module implements a simple feedforward neural network (fully connected) using numpy. It includes layers for fully connected (FC) neurons, activation functions (tanh), and a network class for training and analysis.

### Classes and Functions
- **tanh(x):** Computes the hyperbolic tangent of x.
- **tanh_prime(x):** Computes the derivative of tanh.
- **mse(y_returned, y_expected):** Computes the mean squared error between predicted and expected outputs.
- **mse_prime(y_returned, y_expected):** Computes the derivative of the mean squared error.
- **Layer:** Abstract base class for network layers.
- **FC:** Fully connected layer class with forward and backward propagation methods.
- **Activation:** Activation layer class applying tanh activation and its derivative.
- **Network:** Main class managing layers, loss function, training, and prediction.

### Usage
- Create an instance of `Network`.
- Add layers (FC and Activation) using `addLayer`.
- Set loss function using `setLossFn`.
- Train the network with `train` method.
- Analyze input data using `analyze` method.

---

# solve_xor.py
This script demonstrates the use of the fully connected neural network (`fully_connected.py`) to solve the XOR problem using numpy and Pygame for visualization.

### Execution
- Initializes XOR data (`data_in` and `data_out`).
- Creates a `Network` instance (`fcnn`).
- Adds layers for input, hidden, and output using tanh activation.
- Sets mean squared error as the loss function.
- Trains the network on XOR data for 5000 epochs with a learning rate of 0.05.
- Analyzes the test data (`test_data`) to predict XOR results.

## Notes
- Modify or extend `fully_connected.py` and `solve_xor.py` for different neural network architectures or problems.

## Author
- Richard Christopher 2023
