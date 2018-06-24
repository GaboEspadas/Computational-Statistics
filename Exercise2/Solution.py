"""

Series 2 More Simple Linear Regression and Selection
Author: Gabriel Espadas

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import requests
import pandas as pd
import statsmodels.api as sm

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
plt.scatter(pred, res)
plt.title("Tukey-Ascombe plot")
plt.show()


def cp(model, sigma_squared, n):
    # the Mallows Cp statistic
    # model is of the type produced by statsmodels
    # sigma_squared is the one estimated from the 'full model'
    # n the number of observations
    return model.ssr / sigma_squared - n + 2 * model.df_model


def backward_elimination(X, y):
    # First build the complete model
    full_model = sm.OLS(y, sm.add_constant(X)).fit()
    p = len(X.columns)
    n = len(y)
    full_ssr = full_model.ssr
    sigma_squared = full_ssr / (n - p)
    dict_models = {}
    for i in reversed(range(p)):
        # Look up for the variable to exclude, i.e. the one that increased the ssr the least
        inter_dict = {}
        for j in range(i):
            temp_mod = sm.OLS(y, sm.add_constant(X[:, -j])).fit()
            inter_dict[j] = temp_mod.ssr - full_ssr
        index_to_exclude = min(temp_mod, key=temp_mod.get)
        # fit the model
        model_i = sm.OLS(y,sm.add_constant(X[:,-index_to_exclude])).fit()
         # calculate the cp mallow and save it into the dict
        cpM = cp(model_i,sigma_squared,n)
        dict_models[i]=(index_to_exclude,cpM)

    # retrieve the model with the minimum cp
    excluded_indexes = min(dict_models,key=dict_models.get[1])

# Stepwise Variable Selection: Sklearn does not have this feature!! -> Maybe implement it myself
def stepwise_selection(X, y, type):
    # hard-code thresholds for easyness
    low_trh = 0.01
    upper_trh = 0.05
