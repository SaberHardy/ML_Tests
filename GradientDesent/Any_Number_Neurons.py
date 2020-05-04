import numpy as np


def sigmoid_fun(sop):
    return 1/(1+np.exp(-1*sop))

def error(predict,target):
    return np.power(predict-target,2)

def error_predicted_deriv(predict,target):
    return 2*(predict-target)

def sigmond_sop_derivative(sop):
    return sigmoid_fun(sop) * (1- sigmoid_fun(sop))

def sop_w_dirivative(x):
    return x

def update_w(w,grad,learning_rate):
    return w - learning_rate * grad
#this x is a vector
x = np.array([0.1, 0.4, 4.1])
#the target is our goal
target = np.array([0.55])
learning_rate = 0.001

#np.random.rand it's array contain  float numbers between 0-1
w1_3 = np.random.rand(3)
w2_3 = np.random.rand(3)
w3_3 = np.random.rand(3)
w4_3 = np.random.rand(3)
w5_3 = np.random.rand(3)
#type is an array [ contain 5 random numbers ]
w6_5 = np.random.rand(5)

w6_5_old = w6_5
print("Initial W : ", w1_3,"\n", w2_3,"\n", w3_3,"\n", w4_3,"\n", w5_3,"\n", w6_5)


#Forward_pass
#Hidden_layer_calculation
for k in range(8000):
    sop1 = np.sum(w1_3*x)
    sop2 = np.sum(w2_3*x)
    sop3 = np.sum(w3_3*x)
    sop4 = np.sum(w4_3*x)
    sop5 = np.sum(w5_3*x)

    sig1 = sigmoid_fun(sop1)
    sig2 = sigmoid_fun(sop2)
    sig3 = sigmoid_fun(sop3)
    sig4 = sigmoid_fun(sop4)
    sig5 = sigmoid_fun(sop5)

    #calculate the output layer
    sop_output = np.sum(w6_5 * np.array([sig1, sig2, sig3, sig4, sig5]))

    predicted = sigmoid_fun(sop_output)
    err = error(predicted,target)

    print(f"error result= {predicted}")
    # Backward Pass
    #for every g1,g2,g3 = [0. 0. 0. 0. 0.]

    g1 = error_predicted_deriv(predicted,target)
    g2 = sigmond_sop_derivative(sop_output)
    g3 = np.zeros(w6_5.shape[0])

    #we use this array to replace all the numbers in the array
    g3[0] = sop_w_dirivative(sig1)
    g3[1] = sop_w_dirivative(sig2)
    g3[2] = sop_w_dirivative(sig3)
    g3[3] = sop_w_dirivative(sig4)
    g3[4] = sop_w_dirivative(sig5)

    #type(grad_hidden_output)=array
    grad_hidden_output = g3 * g2 * g1

    w6_5 = update_w(w6_5,grad_hidden_output,learning_rate)
    #this step is for Working with weights between input and hidden layer
    # First Hidden Neuron
    g3 = sop_w_dirivative(w6_5_old[0])
    g4 = sigmond_sop_derivative(sop1)
    g5 = sop_w_dirivative(x)
    grad_hidden_input = g5 * g4 * g3 * g2 * g1
    w1_3 = update_w(w1_3,grad_hidden_input,learning_rate)

    # Second Hidden Neuron
    g3 = sop_w_dirivative(w6_5_old[1])
    g4 = sigmond_sop_derivative(sop2)
    g5 = sop_w_dirivative(x)
    grad_hidden2_input = g5 * g4 * g3 * g2 * g1
    w2_3 = update_w(w2_3,grad_hidden_input,learning_rate)

    # Third Hidden Neuron
    g3 = sop_w_dirivative(w6_5_old[2])
    g4 = sigmond_sop_derivative(sop3)
    g5 = sop_w_dirivative(x)
    grad_hidden3_input = g5 * g4 * g3 * g2 * g1
    w3_3 = update_w(w3_3, grad_hidden3_input, learning_rate)

    # Third Hidden Neuron
    g3 = sop_w_dirivative(w6_5_old[3])
    g4 = sigmond_sop_derivative(sop4)
    g5 = sop_w_dirivative(x)
    grad_hidden4_input = g5 * g4 * g3 * g2 * g1
    w4_3 = update_w(w4_3, grad_hidden4_input, learning_rate)

    # Fourth Hidden Neuron
    g3 = sop_w_dirivative(w6_5_old[4])
    g4 = sigmond_sop_derivative(sop5)
    g5 = sop_w_dirivative(x)
    grad_hidden5_input = g5 * g4 * g3 * g2 * g1
    w5_3 = update_w(w5_3, grad_hidden5_input, learning_rate)

    w6_5_old = w6_5
    print("result predicted is: ",predicted)





