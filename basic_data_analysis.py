import pandas
import os

data = pandas.read_csv("../datasets/titanic/titanic3.csv")
# print(data.head())
# print(data.tail())

# Rows and columns in the dataset
print(data.shape)

print(data.columns.values)

# Basic statistical analysis for each column of the dataset
print(data.describe())

# Types of data for each column
print(data.dtypes)

# Obtain the values per row that present NaN
print(pandas.isnull(data["body"]))
print(pandas.notnull(data["body"]))

# Values() turns it into an array, ravel() create a single lined array that allows
# us to sum the values of True
print(pandas.isnull(data["body"]).values.ravel().sum())
