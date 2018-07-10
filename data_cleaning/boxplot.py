import pandas
import matplotlib.pyplot as plt
import numpy as np
data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")
# print(data.head())

IQR = data["Day Calls"].quantile(0.75)-data["Day Calls"].quantile(0.25)
lower_boxplot_limit = data["Day Calls"].quantile(0.25) - 1.5*IQR
upper_boxplot_limit = data["Day Calls"].quantile(0.75) + 1.5*IQR

print(data["Day Calls"].describe())
print(IQR)
print(lower_boxplot_limit)
print(upper_boxplot_limipr)

plt.boxplot(data["Day Calls"])
plt.ylabel("Número de llamadas diarias")
plt.title("Boxplot del número de llamadas por día")
plt.show()
