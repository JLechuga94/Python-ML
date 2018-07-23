from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import pandas
import numpy as np

data = pandas.read_csv("../../datasets/ads/Advertising.csv")

# Prediction columns with prediction variables
feature_cols = ["TV", "Radio", "Newspaper"]

# We separete the dataset into two, being the prediction variables and the desired
# predicted value
X = data[feature_cols]
Y = data["Sales"]

# With this we can obtain which variables are most important for the modelling
# and the importance of each one when fitting the model to a dataset
# With this information we can better approach the modelling for the dataset
estimator = SVR(kernel="linear")

# We determine the amount of variables we want in the model
selector = RFE(estimator, 2, step=1)
selector = selector.fit(X, Y)
print(selector.support_)
print(selector.ranking_)

X_pred = X[["TV", "Radio"]]
lm = LinearRegression()
lm.fit(X_pred, Y)
print(lm.intercept_)
print(lm.coef_)
print(lm.score(X_pred, Y)) # R^2 value
