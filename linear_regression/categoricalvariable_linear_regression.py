import pandas
import numpy as np
from sklearn.linear_model import LinearRegression

# Since we have 3 categorical variables we need to create two dummy variables
df = pandas.read_csv("../../datasets/ecom-expense/Ecom Expense.csv")
column_names = df.columns.values.tolist()

dummy_gender = pandas.get_dummies(df["Gender"], prefix="Gender")
dummy_city_tier = pandas.get_dummies(df["City Tier"], prefix="City")

# We have to make a data join to add the dummy variables into the original dataset

df_new = df[column_names].join(dummy_gender)
column_names = df_new.columns.values.tolist()

df_new = df_new[column_names].join(dummy_city_tier)


feature_cols = ["Monthly Income", "Transaction Time", "Gender_Male",
    "Gender_Female", "City_Tier 1", "City_Tier 2", "City_Tier 3", "Record"]

X = df_new[feature_cols]
Y = df_new["Total Spend"]

lm = LinearRegression()
lm.fit(X, Y)
# Model obtained with best output for R^2
# Total_spend = 'Monthly Income' * 0.14753898049205721 + 'Transaction Time' * 0.1549461254958953
# 'Gender_Male' * 131.02501325554584 + 'Gender_Female' * -131.02501325554564
# 'City_Tier 1' * 76.76432601049557 + 'City_Tier 2' * 55.13897430923219
# 'City_Tier 3' * -131.9033003197276 + 'Record' * 772.2334457445637
# print(lm.intercept_)
# print(lm.coef_)
# print(list(zip(feature_cols, lm.coef_)))
# print(lm.score(X, Y))

df_new["Prediction"] = df_new['Monthly Income'] * 0.14753898049205721 + df_new['Transaction Time'] * 0.1549461254958953 +  df_new['Gender_Male'] * 131.02501325554584 + df_new['Gender_Female'] * -131.02501325554564 + df_new['City_Tier 1'] * 76.76432601049557 + df_new['City_Tier 2'] * 55.13897430923219 + df_new['City_Tier 3'] * -131.9033003197276 + df_new['Record'] * 772.2334457445637
SSD = np.sum((df_new["Prediction"] -  df_new["Total Spend"]) ** 2)
print(SSD)

RSE = np.sqrt(SSD/(len(df_new)-len(feature_cols)-1)) # Desviación de los valores sobre la media
spend_mean = np.mean(df_new["Total Spend"])
print(RSE)
error = RSE/spend_mean
print(error)
# print(df_new.head())
