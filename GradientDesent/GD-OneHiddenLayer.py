import numpy as np


def sigmoid_function(sop):
    return 1.0 / (1 + np.exp(-1 * sop))


def error_function(predicted, target):
    return np.power(predicted - target, 2)


def error_predicted_deriv(predicted, target):
    return 2 * (predicted - target)


def sigmoid_sop_deriv(sop):
    return sigmoid_function(sop) * (1.0 - sigmoid_function(sop))


def sop_w_deriv(x):
    return x


def update_w(w, grad, learning_rate):
    return w - learning_rate * grad

#inputs and the output are prepared using these 2 lines:
x = np.array([0.1, 0.4, 4.1])
target = np.array([0.2])

learning_rate = 0.001
""" w1_3 is an array holding the 3 weights connecting 
the 3 inputs to the first hidden neuron. 
w2_3 is an array holding the 3 weights connecting the 3 inputs
 to the second hidden neuron"""
w1_3 = np.random.rand(3)
w2_3 = np.random.rand(3)

"""
w3_2 is an array with 2 weights which are for the connections
between the hidden layer neurons and the output neuron.
"""
w3_2 = np.random.rand(2)

w3_2_old = w3_2
print("Initial W : ", w1_3, w2_3, w3_2)

# Forward Pass
# Hidden Layer Calculations
sop1 = np.sum(w1_3 * x)
sop2 = np.sum(w2_3 * x)

sig1 = sigmoid_function(sop1)
sig2 = sigmoid_function(sop2)

# Output Layer Calculations
sop3 = np.sum(w3_2 * np.array([sig1, sig2]))

predicted = sigmoid_function(sop3)
err = error_function(predicted, target)


# Backward Pass
g1 = error_predicted_deriv(predicted, target)

### Working with weights between hidden and output layer
g2 = sigmoid_sop_deriv(sop3)

g3 = np.zeros(w3_2.shape[0])
g3[0] = sop_w_deriv(sig1)
g3[1] = sop_w_deriv(sig2)

grad_hidden_output = g3 * g2 * g1

w3_2 = update_w(w3_2, grad_hidden_output, learning_rate)

### Working with weights between input and hidden layer
# First Hidden Neuron
g3 = sop_w_deriv(w3_2_old[0])
g4 = sigmoid_sop_deriv(sop1)

g5 = sop_w_deriv(x)

grad_hidden1_input = g5 * g4 * g3 * g2 * g1

w1_3 = update_w(w1_3, grad_hidden1_input, learning_rate)

# Second Hidden Neuron
g3 = sop_w_deriv(w3_2_old[1])
g4 = sigmoid_sop_deriv(sop2)

g5 = sop_w_deriv(x)

grad_hidden2_input = g5 * g4 * g3 * g2 * g1

w2_3 = update_w(w2_3, grad_hidden2_input, learning_rate)

w3_2_old = w3_2
print(predicted)