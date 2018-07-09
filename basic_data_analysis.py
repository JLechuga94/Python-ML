import pandas
import os

data = pandas.read_csv("../datasets/titanic/titanic3.csv")
print(data.head())
print(data.tail())

# Rows and columns in the dataset
print(data.shape)

print(data.columns.values)

# Basic statistical analysis for each column of the dataset
print(data.describe())

# Types of data for each column
print(data.dtypes)
