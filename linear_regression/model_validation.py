import pandas
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Divide the dataset into training and testing
data = pandas.read_csv("../../datasets/ads/Advertising.csv")

division_percentage = 0.8
a = np.random.randn(len(data))
check = (a < division_percentage)
training = data[check]
testing = data[~check]

# We are using the model characteristics that presented the best fit in the
# previous exercises
# Sales = 2.966553 + 0.045774 * TV + 0.185021 * Radio
lm = smf.ols(formula='Sales ~ TV + Radio', data = training).fit()
print(lm.params)
print(lm.summary())

sales_pred = lm.predict(testing)
SSD = sum((testing["Sales"]-sales_pred)**2)
print(SSD)
RSE = np.sqrt(SSD/(len(testing)-2-1))
print(RSE)
sales_mean = np.mean(testing["Sales"])
mean_error = RSE/sales_mean
print(mean_error)
