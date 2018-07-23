import pandas
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

data = pandas.read_csv("../../datasets/ads/Advertising.csv")
# Simple linear regression with one variable
lm = smf.ols(formula='Sales ~ TV', data = data).fit()
# Sales = 7.032594 + 0.047537  * TV_expenses


print(lm.params)
# print(lm.pvalues)
# print(lm.rsquared) # Adjustment level between dataset and model
# print(lm.rsquared_adj)
# print(lm.summary())


tv_df = pandas.DataFrame(data["TV"])
# We need to pass a data frame to make the prediction
sales_pred = lm.predict(tv_df)
# print(sales_pred) # Predicted values for the model

data.plot(kind="scatter", x = "TV", y = "Sales")
plt.plot(tv_df, sales_pred, c="red", linewidth=2)
plt.show()

# Standard deviation calculation for analysis
data["sales_pred"] = 7.02594 + 0.047537*data["TV"]
data["RSE"] = (data["Sales"]-data["sales_pred"])**2
SSD = sum(data["RSE"])
RSE = np.sqrt(SSD/(len(data)-2))
print(RSE) # STD in sales

# The error between the mean value of sales and the RSE of predicted and observed
# sales values. The percentage of error represented by the dataset'??
sales_mean = np.mean(data["Sales"])
mean_error = RSE/sales_mean
print(mean_error)
