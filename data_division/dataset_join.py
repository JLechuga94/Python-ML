import pandas

filepath = "../../datasets/athletes/"

# Medals won by athlete in different Olympic games
data_main = pandas.read_csv(filepath + "Medals.csv", encoding="ISO-8859-1")
print(data_main.head())
a = data_main["Athlete"].unique().tolist()  # We want non-repeated athletes
print(len(a))  # Unique athletes
print(data_main.shape)  # Size of dataset

# Country of origin of athletes
data_country = pandas.read_csv(filepath + "Athelete_Country_Map.csv", encoding="ISO-8859-1")
print(data_country.head())
print(len(data_country))

print(data_country[data_country["Athlete"] == "Aleksandar Ciric"])  # Repeated athletes for different countries
data_country = data_country.drop_duplicates(subset="Athlete")  # This will only leave one country for each athlete

# We compare against the list of unique athletes to check the length of datasets to be equal
print(len(data_country) == len(a))

# Sports performed by each athlete
data_sports = pandas.read_csv(filepath + "Athelete_Sports_Map.csv", encoding="ISO-8859-1")
print(data_sports.head())
print(len(data_sports))  # Sports performed per player

# Repeated athletes for different sports which makes the length different from the other datasets
print(data_sports[
    (data_sports["Athlete"] == "Chen Jing") |
    (data_sports["Athlete"] == "Richard Thompson") |
    (data_sports["Athlete"] == "Matt Ryan")]
    )
data_sports = data_sports.drop_duplicates(subset="Athlete")  # This will only leave one sport for each athlete

# Merge of two dataset tables based on the athlete and their country
data_main_country = pandas.merge(
    left=data_main, right=data_country,
    left_on="Athlete", right_on="Athlete"
    )
print(data_main_country.head())
print(data_main_country.shape)  # Size of the list
print(data_main_country[data_main_country["Athlete"] == "Aleksandar Ciric"])

data_final = pandas.merge(
    left=data_main_country, right=data_sports,
    left_on="Athlete", right_on="Athlete"
)
print(data_final.head())
