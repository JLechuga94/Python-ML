from IPython.display import Image
import numpy as np
import pandas

filepath = "../../datasets/athletes/"

# Medals won by athlete in different Olympic games
data_main = pandas.read_csv(filepath + "Medals.csv", encoding="ISO-8859-1")
data_country = pandas.read_csv(filepath + "Athelete_Country_Map.csv", encoding="ISO-8859-1")
data_country = data_country.drop_duplicates(subset="Athlete")  # This will only leave one country for each athlete
data_sports = pandas.read_csv(filepath + "Athelete_Sports_Map.csv", encoding="ISO-8859-1")
data_sports = data_sports.drop_duplicates(subset="Athlete")  # This will only leave one sport for each athlete

# List of athletes to be eliminated from dataset
out_athlete = np.random.choice(data_main["Athlete"], size=6, replace=False)

# Elimination of Michael Phelps and random 6 athletes from the dataset
data_country_dlt = data_country[(-data_country["Athlete"].isin(out_athlete)) & (data_country["Athlete"] != "Michael Phelps")]

data_sports_dlt = data_sports[(-data_sports["Athlete"].isin(out_athlete)) & (data_sports["Athlete"] != "Michael Phelps")]

data_main_dlt = data_main[(-data_main["Athlete"].isin(out_athlete)) & (data_main["Athlete"] != "Michael Phelps")]

# print(data_country_dlt.head())
# print(len(data_country_dlt))
# print(len(data_sports_dlt))
# print(len(data_main_dlt))

# Inner joind between a dataset that has all the info and another missing data
# 7 athletes

merged_inner = pandas.merge(left=data_main, right=data_country_dlt, how='inner', left_on="Athlete", right_on="Athlete")
print(len(merged_inner))

merged_left = pandas.merge(left=data_main, right=data_country_dlt, how='left', left_on="Athlete", right_on="Athlete")
print(len(merged_left))

merged_right = pandas.merge(left=data_main_dlt, right=data_country, how='right', left_on="Athlete", right_on="Athlete")
print(len(merged_right))
print(merged_right.tail(10))

data_country_appended = data_country_dlt.append(
    {
        "Athlete": "Julian Lechuga",
        "Country": "Mexico"
    }, ignore_index=True
)
merged_outer = pandas.merge(left=data_main, right=data_country_appended, how='outer', left_on="Athlete", right_on="Athlete")
print(len(merged_outer))
print(merged_outer.head())
print(merged_outer.tail())
print(data_country_appended.tail())
