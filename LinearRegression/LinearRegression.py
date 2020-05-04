# import matplotlib.pyplot as plt
# from scipy import stats
#
# x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
# y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
# slope,intercept,r,p,std_err=stats.linregress(x,y)
#
# #Create a function that uses the slope and intercept values
# # to return a new value. This new value represents where
# # on the y-axis the corresponding x value will be placed:
# def myfun(x):
#     return slope * x + intercept
#
# #Run each value of the x array through the function.
# # This will result in a new array with new values for the y-axis:
# mymodel = list(map(myfun,x))
# # print(mymodel)
# # spped = myfun(10)
#
# #Draw the original scatter plot:
# plt.scatter(x,y)
# #Draw the line of linear regression:
# plt.plot(x,mymodel)
# #Display the diagram:
# plt.show()
#-----------------------------------------------------------
#the next code is about prediction of future values

# from scipy import stats
# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
# slope,intercept,r,p,std_err = stats.linregress(x,y)
# print(r) # r = -0.76
# #The result -0,76 shows that there is a relationship,
# # not perfect, but it indicates that we could use
# # linear regression in future predictions.
#
#Predict Future Values
# def myfunc(x):
#   return slope * x + intercept
#
# speed = myfunc(10)
#
# print(speed)
#---------------------------------------------------------
# import matplotlib.pyplot as plt
# from scipy import stats
#
# x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
# y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
# # x = [1995,1997,1999,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019]
# # y = [0.5,1.0,1.5,1.5,1.5,1.5,2.0,2.5,2.0,2.5,2.7,2.8,2.0]
#
#
# slope, intercept, r, p, std_err = stats.linregress(x, y)
# print(r)
# def myfunc(x):
#   return slope * x + intercept
#
# mymodel = list(map(myfunc, x))
# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()
#----------------------------------------------------------

#third example
#Linear Regression on Boston Housing Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

# %matplotlib inline
#load data from scikit-learn
from sklearn.datasets import load_boston
boston_dataset = load_boston()
#boston_dataset.keys(): it shows to us the first row name of columns
# print(boston_dataset.keys())
#boston_dataset.DESC():The description of all the feature
# print(boston_dataset.DESCR)
boston = pd.DataFrame(boston_dataset.data,columns=boston_dataset.feature_names)
# print(boston.head())
"""
create new column of target values and add it to the dataframe
"""
boston['MEDV'] = boston_dataset.target
# print(boston['MEDV'])
print(boston.isnull().sum())
#Exploratory Data Analysis
#visalusation the data to undestand
sb.set(rc={'figure.figsize':(11.7,8.27)})
sb.distplot(boston['MEDV'], bins=30)
plt.show()

correlation_matrix = boston.corr().round(2)
sb.heatmap(data=correlation_matrix,annot=True)
# print("corr=:",correlation_matrix)

#Based on the above observations we will RM and LSTAT as our features.
plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    plt.show()
















