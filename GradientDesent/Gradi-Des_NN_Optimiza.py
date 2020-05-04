# import numpy as np
#
# #this example is for ( 1 Input – 1 Output )
# #Forward Pass
# def sigmoid(sop):
#     #SOP that's mean Sum Of Products
#     return 1/(1+np.exp(-1*sop))
#
# def error(predected,target):
#     return np.power(predected-target,2)
#
# #Backward Pass
#
# def activation_sop_derivative(sop):
#     return sigmoid(sop) * (1 - sigmoid(sop))
#
# def error_prediction_diretive(predict,target):
#     return 2*(predict - target)
#
# def sop_w_deriv(x):
#     return x
#
# def update_w(w,grad,learning_rate):
#     return w - learning_rate * grad
# #input x
# x1 = 0.1
# x2 = 0.4
# target = 0.3
# learning_rate = 0.1
# """
# The weight is initialized randomly using numpy.random.rand()
# which returns a number between 0 and 1
# """
# w1 = np.random.rand()
# w2 = np.random.rand()
# print(f"Initial Weight:{w1} and {w2} ")
#
#
# # Forward Pass
# y = w1 * x1 + w2*x2
# print("y=",y)
# predicted = sigmoid(y)
# err = error(predicted, target)
# print("error= ",err)
#
#
# # Backward Pass
# """
# the error is calculated using 2 terms which are:
#     predicted
#     target
# """
# g1 = error_prediction_diretive(predicted, target)
# print('g1 = ',g1)
# g2 = activation_sop_derivative(predicted)
# print('g2 = ',g2)
# g3w1 = sop_w_deriv(x1)
# g3w2 = sop_w_deriv(x2)
#
#
# #This returns the gradient by which the weight value could be updated
# """
# Derror/Dw = (Derror/Dpredict)*(Dpredict/Dsop)*(Dsop/Dw)
# """
# gradw1 = g3w1 * g2 * g1
# gradw2 = g3w2 * g2 * g1
# print('gradw1 = ',gradw1)
# print('gradw2 = ',gradw2)
#
# print("predicted=",predicted)
#
# second_w1 = update_w(w1, gradw1, learning_rate)
# second_w2 = update_w(w2, gradw2, learning_rate)
# print(f"the weight of the input is--> {second_w1} and {second_w2}",)
#
# #this is just test
# def virefication():
#     if round(second_w1,2) == round(w1,2) \
#             and round(second_w2,2) == round(w2,2):
#         return print(True)
#     else:
#         return print(False)
#
# virefication()
#
#this exmple it's describe the gradient-descent for 10 inputs
import numpy as np

def sigmoid(sop):
    return 1.0/(1+np.exp(-1*sop))

def error(predicted, target):
    return np.power(predicted-target, 2)

def error_predicted_deriv(predicted, target):
    return 2*(predicted-target)

def predictive_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))

def sop_w_deriv(x):
    return x

def update_w(w, grad, learning_rate):
    return w - learning_rate*grad

x = np.array([0.1, 0.4, 1.1, 1.3, 1.8, 2.0, 0.01, 0.9, 0.8, 1.6])
target = np.array([0.2])

learning_rate = 0.1
#every time we initialize the w with random.rand
w = np.random.rand(10)
print("weight = ",w)

for k in range(1000):
    #Forward Pass
    y = np.sum(w*x)
    predicted = sigmoid(y)
    err = error(predicted,target)
    print(f"the error value is: {err}")
    g1 = error_predicted_deriv(predicted,target)
    g2 = predictive_sop_deriv(y)
    g3 = sop_w_deriv(x)

    grad = g3 * g2 * g1
    w = update_w(w, grad, learning_rate)
    print(f"the predicted value is: {predicted}")

