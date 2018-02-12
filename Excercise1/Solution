"""

Series 1 Simple Linear Regression
Author: Gabriel Espadas

"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import requests

## 2 - a)
# Randomly generate 100 times a vector of y-values according
# to the above model with the given x-values and generates a vector of slopes of the regression lines.

# Go for something simpler, the sklearn
x_train = np.arange(1, 41).reshape(40, 1)
n_samples = x_train.shape[0]
slopes = []
linear_model = LinearRegression()

for i in range(100):
    y_train = 2 * x_train + 1 + np.random.normal(0, scale=5, size=n_samples).reshape((40, 1))
    linear_model.fit(X=x_train, y=y_train)
    slopes.append(linear_model.coef_[0][0])

# 2 -b) Generate a histogram of the slopes together with the theoretical density of the slope estimator
mu = 2
extended_x = np.concatenate((np.ones(n_samples).reshape(40, 1), x_train), axis=1)
xt_x = np.dot(extended_x.transpose(), extended_x)
sigma = 5 * np.sqrt(np.linalg.inv(xt_x)[1, 1])
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(np.asanyarray(slopes), density=True)
ax1.plot(x, mlab.normpdf(x, mu, sigma))
ax1.set_title("Hist of Slope")

# 2 - c) Compute the mean and empirical deviation of the estimated slopes
mu_bar, sigma_bar = np.mean(np.asanyarray(slopes)), np.std(np.asanyarray(slopes))
print("theoretical mean: " + str(mu) + ", theoretical standard dev: " + str(sigma))
print("empirical mean: " + str(mu_bar) + ", empirical standard dev: " + str(sigma_bar))

# 2 - d) Repeat b  with a chi squared dist

slopes_chi = []
for i in range(100):
    y_train = 2 * x_train + 1 + 5 * ((1 - np.random.chisquare(df=1, size=n_samples)) / np.sqrt(2)).reshape((40, 1))
    linear_model.fit(X=x_train, y=y_train)
    slopes_chi.append(linear_model.coef_[0][0])

ax2.hist(np.asanyarray(slopes_chi), density=True)
ax2.plot(x, mlab.normpdf(x, mu, sigma))
ax2.set_title("Hist of Chi-Slope")
plt.show()
# 3 -a) Fetch data, plot it and describe it

data_airlines = requests.get("http://stat.ethz.ch/Teaching/Datasets/airline.dat").text
data_airlines = data_airlines.split()
data_airlines = np.asanyarray(data_airlines).astype(int)
plt.figure()
plt.plot(data_airlines)
plt.title("Original Airline Data")
plt.show()

data_airlines_log = np.log(data_airlines)
plt.figure()
plt.plot(data_airlines_log)
plt.title("Log Airline Data")
plt.show()

linear_comp = np.arange(1, 145).reshape(144, 1)
months_cate = np.zeros((12, 12))
np.fill_diagonal(months_cate, 1)
months_cate = np.tile(months_cate, (12, 1))
X = np.concatenate((linear_comp, months_cate), axis=1)
airlines_linear_model = LinearRegression(fit_intercept=False)
airlines_linear_model.fit(X, data_airlines_log)
y_bar = airlines_linear_model.predict(X)
res = data_airlines_log - y_bar

plt.figure()
plt.plot(data_airlines_log, color='green', marker='.', label='Real')
plt.title("Log-Data vs Fit")
plt.plot(y_bar, color="orange", marker=".", label="Estimated")
plt.show()

plt.figure()
plt.plot(res,color = "red",label = "Residual")
plt.title("Residuals Airlines Linear Model")
plt.show()
