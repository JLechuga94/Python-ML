import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

data = pandas.read_csv("../../datasets/auto/auto-mpg.csv")
def regression_validation(X_data, Y, Y_predict):
    SSD = np.sum((Y - Y_predict)**2)
    RSE = np.sqrt(SSD/(len(X_data) - 1))
    y_mean = np.mean(Y)
    error = RSE/y_mean
    print("SSD: " + str(SSD), "RSE: " + str(RSE), "Y mean: " + str(y_mean), "Error: " + str(error*100))
# We want to eliminate the elements that are NaN in the dataset
data["mpg"] = data["mpg"].fillna(data["mpg"].mean())
data["horsepower"] = data["horsepower"].fillna(data["horsepower"].mean())
X = data["horsepower"]
Y = data["mpg"]
lm = LinearRegression()
# plt.plot(data["horsepower"], data["mpg"], "ro")
# plt.xlabel("Horsepower")
# plt.ylabel("Miles per gallon consumption")
# plt.show()

# Linear regression model----------------------------------------------------
# mpg = a + b*horsepower ----> 0.5746533406450252 R^2 score
# X_data = X[:, np.newaxis]


# Quadratic regression model -----------------------------------------------
# mpg = a + b * horsepower^2 ---> 0.4849887034823205 R^2 score
# X_data = X**2
# X_data = X_data[:, np.newaxis]

# Linear and quadratic regression model -----------------------------------------------
# mpg = a + b * horsepower + c * horsepower^2 ---> 0.6439066584257469 R^2 score
# mpg = 55.02619244708032 -0.43404318 * hp + 0.00112615* hp^2
poly = PolynomialFeatures(degree=2) # Generate a polynomial of the degree required
X_data = poly.fit_transform(X[:, np.newaxis]) # Fit the polynomial to the x values
lm = linear_model.LinearRegression()

# FOR cycle to iteration over different polynomials of different degree

for degree in range(2,6):
    poly = PolynomialFeatures(degree=degree) # Generate a polynomial of the degree required
    X_data = poly.fit_transform(X[:, np.newaxis]) # Fit the polynomial to the x values
    lm = linear_model.LinearRegression()
    lm.fit(X_data, Y)
    print("Polynomial regression of degree " + str(degree))
    print("R^2 score: " + str(lm.score(X_data, Y)*100)) # R^2 score value
    print("Model intercept " + str(lm.intercept_))
    print("Model coefficients " + str(lm.coef_))
    regression_validation(X_data, Y, lm.predict(X_data))
    print("\n")



# lm.fit(X_data, Y)
# plt.plot(X_data, Y, "ro")
# plt.plot(X_data, lm.predict(X_data), color="blue")
# plt.show()
# print(lm.score(X_data, Y)) # R^2 score value
