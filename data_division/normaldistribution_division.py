# Divition of dataset for training and testing
import pandas
import numpy as np
import matplotlib.pyplot as plt

division_percentage = 0.75
data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")
# print(data.head())

# Division using normal distribution

a = np.random.randn(len(data))
# plt.hist(a)

check = (a < division_percentage)

plt.hist(check)

training_dataset = data[check]
testing_dataset = data[~check]

# Amount of elements for each division of the DS
print(len(training_dataset))
print(len(testing_dataset))


plt.show()
