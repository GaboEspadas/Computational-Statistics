"""

Series 2 More Simple Linear Regression and Selection
Author: Gabriel Espadas

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import pandas as pd

# 1 - a) Download data, exploratory analysis, linear regression and residual analysis

# Exploratory
url = "http://stat.ethz.ch/Teaching/Datasets/mortality.csv"
pollution_data = pd.read_csv(url, sep=",")
pollution_data.set_index("City", inplace=True)
pd.plotting.scatter_matrix(pollution_data, figsize=(15, 15), hist_kwds={'bins': 20}, s=60, alpha=.5)

# Linear Model
linear_model = LinearRegression()
X = pollution_data.iloc[:, 1:]
y = pollution_data.iloc[:, 0]
linear_model.fit(X, y)
pred = linear_model.predict(X)
res = y - pred
plt.scatter(pred,res)
plt.title("Tukey-Ascombe plot")
plt.show()

# Stepwise Variable Selection: Sklearn does not have this feature!! -> Maybe implement it myself