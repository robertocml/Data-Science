# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 21:20:10 2019

@author: Roberto Perez Barrera A01380452
Referencias:
https://stackoverflow.com/questions/17784587/gradient-descent-using-python-and-numpy
"""


import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt  


def gradientDescent(x, y, theta, alpha, m, nt):
    xTrans = x.transpose()
    for i in range(0, nt):
        hypothesis = np.dot(x,theta)
        loss = hypothesis - y
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta


x = pd.read_csv('casas1.txt', sep=",", header=None, usecols=[0])
y = pd.read_csv('casas1.txt', sep=",", header=None, usecols=[1])

m,n = np.shape(x)
bias_vector = np.ones((m,1))
x_b = np.hstack((bias_vector,x))
m,n = np.shape(x_b)

y = np.squeeze(np.asarray(y))

numIterations= 1000
alpha = 0.01
theta = np.ones(n)
theta = gradientDescent(x_b, y, theta, alpha, m, numIterations)
print(theta)

prediction = np.dot(x_b, theta)
 

plt.plot(x, prediction, "r--")  
plt.plot(x, y, "b.")  
plt.axis([4, 23, -5, 25])
plt.show()