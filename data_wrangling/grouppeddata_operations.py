import pandas
import numpy as np

n = 500

gender = ["Male", "Female"]
income = ["Poor", "Middle Class", "Rich"]
height = np.ceil(160 + 30 * np.random.randn(n))
weight = np.ceil(65 + 25 * np.random.randn(n))
age = np.ceil(30 + 12 * np.random.randn(n))
income_value = 18000 + 3500 * np.random.randn(n)
imc = [weight[i]/np.power(height[i]/100, 2) for i in range(0, n)]
gender_data = [np.random.choice(gender) for i in range(0, n)]
income_data = [np.random.choice(income) for i in range(0, n)]

data = pandas.DataFrame(
    {
        "Gender": gender_data,
        "Economic status": income_data,
        "Height": height,
        "Weight": weight,
        "IMC": imc,
        "Age": age,
        "Income": income_value
    }
)
# Groups data based on a specific column of the data set
grouped_gender = data.groupby("Gender")
grouped_status = data.groupby("Economic status")

# Groupping based on two or more categories of the dataset
double_group = data.groupby(["Gender", "Economic status"])
double_grouped_income = double_group["Income"]

agg1 = double_group.aggregate(
    {
        "Income": np.sum,
        "Age": np.mean,
        "Height": np.std
    }
)

# Lambda function for the height
agg2 = double_group.aggregate(
    {
        "Age": np.mean,
        "Height": lambda h: (np.mean(h))/np.std(h)
    }
)

# Apply the aggregate operations to all columns of the group
agg3 = double_group.aggregate([np.sum, np.mean, np.std])

agg4 = double_group.aggregate([lambda x: (np.mean(x))/np.std(x)])

# print(double_group.sum())
# print(double_group.mean())
# print(double_group.size())  # Amount of rows for each condition
# print(double_group.describe())
# print(double_grouped_income.describe())  # Allows for specific description of a single value
# print(agg3)
print(agg4)
