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

# Groupping based on two or more categories of the DS
double_group = data.groupby(["Gender", "Economic status"])
double_grouped_income = double_group["Income"]

# Returns the elements of the group  (1 of 6) in which the sum of the Age value
# is greater than 2400
filt1 = double_group["Age"].filter(lambda x: x.sum() > 2400)
# print(filt1)

# Variable transformation
# zscore = lambda x: (x - x.mean())/x.std()
# z_group = double_group.transform(zscore)
# print(z_group)
# plt.hist(z_group["Age"])
# plt.show()

# Fill the NaN values in the DS column with a specific value
# fill_na_mean = lambda x: (x.fillna(x.mean()))
# double_group.transform(fill_na_mean)

# Very useful operations for grouped data
ex1 = double_group.head(1)  # 1st elements for each group in the grouped DS
ex2 = double_group.tail(1)  # Last elements for each group in the grouped DS
ex3 = double_group.nth(32)  # Nth value for each group in the grouped DS
# print(ex1)
# print(ex2)
# print(ex3)


# Sort the data by values. In this case we sort them by age and income
data_sorted = data.sort_values(["Age", "Income"])
print(data_sorted.head(10))

# We still have the DS sorted by age but it is now grouped by gender and
# we can obtain the youngs/oldest Female or Male in the DS
age_grouped = data_sorted.groupby("Gender")
print(age_grouped.head(1))
