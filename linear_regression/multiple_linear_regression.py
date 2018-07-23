import pandas
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Possible models for the data set following 2^k-1
# Sales - TV
# Sales - Newspaper
# Sales - Radio
# Sales - TV + Newspaper
# Sales - TV + Radio
# Sales - Newspaper + Radio
# Sales - TV + Newspaper + Radio

data = pandas.read_csv("../../datasets/ads/Advertising.csv")

# Basic model fitting to dataset with 1 variable SLR
lm = smf.ols(formula='Sales ~ TV', data = data).fit()
#------------------------------------------------------------------------------
# Model fitting for MLR adding a Newspaper variable
# Sales = 5.774948 + 0.046901 * TV_expenses + 0.044219 * Newspaper_expenses
lm2 = smf.ols(formula='Sales ~ TV + Newspaper', data = data).fit()

# print(lm2.params)
# print(lm2.pvalues)
# print(lm2.rsquared) # Adjustment level between dataset and model
# print(lm2.rsquared_adj)
# print(lm.summary())

# sales_pred = lm2.predict(data[["TV","Newspaper"]])
# SSD = sum((data["Sales"]-sales_pred)**2)
# print(SSD)
# RSE = np.sqrt(SSD/(len(data)-2-1))
# print(RSE)
# sales_mean = np.mean(data["Sales"])
# mean_error = RSE/sales_mean
# print(mean_error)
#------------------------------------------------------------------------------
# Model fitting for MLR adding a Radio variable. Excellent fit!
# Sales = 2.921100 + 0.045755 * TV_expenses + 0.187994 * Radio_expenses
lm3 = smf.ols(formula='Sales ~ TV + Radio', data = data).fit()
print(lm3.params)
# print(lm3.pvalues)
# print(lm3.rsquared) # Adjustment level between dataset and model
# print(lm3.rsquared_adj)
print(lm3.summary())

sales_pred = lm3.predict(data[["TV","Radio"]])
SSD = sum((data["Sales"]-sales_pred)**2)
print(SSD)
RSE = np.sqrt(SSD/(len(data)-2-1))
print(RSE)
sales_mean = np.mean(data["Sales"])
mean_error = RSE/sales_mean
print(mean_error)
#------------------------------------------------------------------------------
# Model fitting for MLR adding all variables. Newspaper not good, suboptim performance
# Sales = 2.9389 + 0.045765 * TV_expenses + 0.188530 * Radio_expenses - 0.001037 * Newspaper_expenses
lm4 = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data = data).fit()
print(lm4.params)
# print(lm4.pvalues)
# print(lm4.rsquared) # Adjustment level between dataset and model
# print(lm4.rsquared_adj)
print(lm4.summary())

sales_pred = lm4.predict(data[["TV","Radio", "Newspaper"]])
SSD = sum((data["Sales"]-sales_pred)**2)
print(SSD)
RSE = np.sqrt(SSD/(len(data)-2-1))
print(RSE)
sales_mean = np.mean(data["Sales"])
mean_error = RSE/sales_mean
print(mean_error)
