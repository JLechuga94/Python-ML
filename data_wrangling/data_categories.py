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
# In this way we can group the data set by gender and obtain specfic data
# related to Male and Female individuals
for names, groups in double_group:
    print(names)
    print(groups)
# print(data.head(20))
print(grouped_gender.groups)
print(grouped_gender.get_group("Female"))
