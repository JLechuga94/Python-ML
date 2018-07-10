import pandas

data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")
# print(data.head())
# print(data[:8])

#Filter the whole data set by conditions on specific columns and rows

# Users with Total Mins > 200
data1 = data[data["Day Mins"] > 200]
# print(data1)

# Users from New York (State = NY)
data2 = data[data["State"] == "NY"]
# print(data2)

data3 = data[(data["State"] == "NY" ) & (data["Day Mins"] > 300)]
# print(data3)
# print(data3.shape) # Only print information of the amount of rows and columns

data4 = data[(data["State"] == "NY" ) | (data["Day Mins"] > 300)]
# print(data4.shape)

data5 = data[data["Night Calls"] > data["Day Calls"]]
print(data5.shape)

data6 = data[data["Night Mins"] > data["Day Mins"]]
print(data6.shape)
