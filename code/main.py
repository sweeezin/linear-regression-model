#import libraries
import matplotlib
matplotlib.use('TkAgg')  # Use an interactive backend

import matplotlib.pyplot as plt

import pandas as pd
import pylab as pl 
import numpy as np

from sklearn import linear_model
regr = linear_model.LinearRegression()

#import dataset

df = pd.read_csv("/home/fionajin/github/linear-regression-model/code/Ecommerce_Customers.csv")
print(df.head(3))

cdf = df[["Avg Session Length", "Time on App", 
               "Time on Website",'Length of Membership',"Yearly Amount Spent"]]

#plotting 
plt.scatter(cdf[["Length of Membership"]], cdf[["Yearly Amount Spent"]],  color='blue')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

msk = np.random.rand(len(df)) < 0.8

train = cdf[msk]
test = cdf[~msk]

#training
plt.scatter(train[["Length of Membership"]], train[["Yearly Amount Spent"]],  color='blue')
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")
plt.show()

inputCols=["Avg Session Length", "Time on App", 
               "Time on Website",'Length of Membership']

x = np.asanyarray(train[inputCols])
y = np.asanyarray(train[['Yearly Amount Spent']])

regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

y_hat= regr.predict(test[inputCols])
x = np.asanyarray(test[inputCols])
y = np.asanyarray(test[['Yearly Amount Spent']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

