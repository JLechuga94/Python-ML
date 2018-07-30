import pandas

data = pandas.read_csv("../../datasets/gender-purchase/Gender Purchase.csv")
print(data.head())
print(data.shape)

# Cross table
contingency_table = pandas.crosstab(data["Gender"], data["Purchase"])
print(contingency_table)

print(contingency_table.sum(axis=1))
