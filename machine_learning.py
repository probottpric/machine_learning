# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:17:30 2020

@author: ProBot
"""

import sklearn
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import model_selection
from sklearn import linear_model
import math as m

x=range(1,90)
y=[m.sin(i/10)*900 for i in x]
z=[i**2 for i in x]

plt.xlabel('x val')
plt.ylabel('y val')
plt.title('Title x-y graph')
#plt.axis([0,200,-1.1,1.1])
plt.grid(True)
plt.plot(x,y,'b--o',linewidth=1,markersize=4,label='sin')
plt.plot(x,z,'r--o',linewidth=1,markersize=4,label='cos')
plt.legend(loc='upper right')
plt.show()
'''
#y=mx+b
#y=1.8m+32

#simple linear graph with rand

x=list(range(0,100)) #C
y=[(1.8*F)+32+random.randint(-5,5) for F in x] #F

#print(x,y)
plt.plot(x,y,'-*r')
plt.show()

x=np.array(x).reshape(-1,1)
y=np.array(y).reshape(-1,1)

xTrain, xTest, yTrain,yTest=model_selection.train_test_split(x,y,test_size=0.2)
model = linear_model.LinearRegression()
model.fit(xTrain,yTrain)
accuracy=model.score(xTest,yTest)

print('coefficient:',model.coef_)
print('Intercept:',model.intercept_)
print('accuracy:',accuracy*100)
#print(xTrain.shape)

print('    peebee')
'''