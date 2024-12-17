#import libraries

#graph/data visualization
import matplotlib as plt 

#data manipulation
import pandas as pd

import pylab as pl 

import numpy as np

from sklearn import linear_model
regr = linear_model.LinearRegression()

#import dataset

df = pd.read_csv("Ecommerce_Customers.csv")
df.head(3)
