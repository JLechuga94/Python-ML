import pandas

data = pandas.read_csv("../datasets/titanic/titanic3.csv")

# axis = 0 means rows, axis = 1 means columns how='all' deletes only if all values
# in the row/column are NaN, how='any' deletes the row/column if any value is NaN
# data.dropna(axis=0, how="all")
# data.dropna(axis=0, how="any")

data3 = data
# fillna Replaces all NaN in the dataset with the input value
data3.fillna(0)

data4 = data
data4.fillna('Unknown')

# In this way we can assign different types of value to different columns
data5 = data
data5["body"] = data5["body"].fillna(0)
data5["home.dest"] = data5["home.dest"].fillna("Unknown")
data5["age"] = data5["age"].fillna(data["age"].mean())
data5["age"] = data5["age"].fillna(method="backfill")

print(data5.head(30))
