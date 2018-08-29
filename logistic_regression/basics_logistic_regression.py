import pandas

data = pandas.read_csv("../../datasets/gender-purchase/Gender Purchase.csv")
print(data.head())
print(data.shape)

# Cross table
contingency_table = pandas.crosstab(data["Gender"], data["Purchase"])
print(contingency_table)

print(contingency_table.sum(axis=1)) # Sum values of rows
print(contingency_table.sum(axis=0)) # Sum values of columns

# Proportion of Men and Women that bought and Men/Women that didnt bought
print(contingency_table.astype("float").div(contingency_table.sum(axis=1), axis=0))
