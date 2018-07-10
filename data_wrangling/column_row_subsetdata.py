import pandas

data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")
# print(data.head())
# print(data[:8])

data1 = data[["Day Mins", "Night Mins", "Account Length"]][:50]
# print(data1)

# Filtering with position based indexing 'iloc' for rows and columns
data2 = data.iloc[:10,:3]
# print(data2)

data3 = data.iloc[:10,:]
# print(data3)

data4 = data.iloc[[1,34,89],[2,5,7]]
# print(data4)

# Filtering by label based indexing 'loc' for rows and columns
data5 = data.loc[[1,34,89],["Area Code","VMail Plan","Day Mins"]]
# print(data5)

#Creation of a new column in the DataFrame based on existing information of the data
data["Total Mins"] = data["Day Mins"] + data["Night Mins"] + data["Eve Mins"]
print(data["Total Mins"])
