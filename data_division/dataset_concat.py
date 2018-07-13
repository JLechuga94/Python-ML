import pandas

red_wine = pandas.read_csv("../../datasets/wine/winequality-red.csv", sep=";")
white_wine = pandas.read_csv("../../datasets/wine/winequality-white.csv", sep=";")
red_wine_columns = red_wine.columns.values
white_wine_columns = white_wine.columns.values

# print(red_wine.head())
# print(red_wine.shape)
# print(white_wine.head())
# print(white_wine.shape)
# print(red_wine_columns)
# print(white_wine_columns)
# print(red_wine_columns == white_wine_columns)

# The data is concatenated in the order that is stated in the array
wine_data = pandas.concat([red_wine, white_wine], axis=0)

print(wine_data.head())
print(wine_data.shape)
