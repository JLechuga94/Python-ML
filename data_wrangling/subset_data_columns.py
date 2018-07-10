import pandas

data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")
# print(data.head())

# Selection of a specific set of data by selection one column

account_length = data["Account Length"]
# print(account_length.head())

# Creation of subset data from several datas
subset = data[["Account Length", "Phone", "Eve Charge", "Day Calls"]]
# print(subset.head())

desired_columns = ["Account Length", "Phone", "Eve Charge", "Day Calls"]
subset = data[desired_columns]
# print(subset.head())

# Simple list comprehension for creating a subset based on the list of desired items
desired_columns = ["Account Length", "VMail Message", "Day Calls"]
all_columns_list = data.columns.values.tolist()

print(all_columns_list)
sublist = [column for column in all_columns_list if column in desired_columns]

print(sublist)
