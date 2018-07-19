import pandas
import numpy as np
import matplotlib.pyplot as plt


data_ads = pandas.read_csv("../../datasets/ads/Advertising.csv")
print(data_ads.head())
data_columns = data_ads.columns.values

# Manual calculation of Pearson correlation


def pearson_coeff(df, var1, var2):
    corrn = (data_ads[var1] - np.mean(data_ads[var1])) * (data_ads[var2] - np.mean(data_ads[var2]))
    corr1 = (data_ads[var1] - np.mean(data_ads[var1])) ** 2
    corrn2 = (data_ads[var2] - np.mean(data_ads[var2])) ** 2
    pearson = sum(corrn)/np.sqrt(sum(corr1) * sum(corrn2))
    return pearson


for x in data_columns:
    for y in data_columns:
        if x != y:
            print(x + ", " + y + ": " + str(pearson_coeff(data_ads, x, y)))

# Pandas function for calculation of Pearson correlation
# Outputs a simetric matrix of correlation
print(data_ads.corr())
plt.matshow(data_ads.corr())  # Graphic representation of the corr matrix
# plt.plot(data_ads["TV"], data_ads["Sales"], "o")
plt.show()
