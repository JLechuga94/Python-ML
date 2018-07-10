import pandas
import matplotlib.pyplot as plt
import numpy as np
data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")
# print(data.head())

# Sturges law to adequate number of divisions for size of sample
k = int(np.ceil(1 + np.log2(3333)))
plt.figure(2)
plt.hist(data["Day Calls"], bins=k)
plt.xlabel("Nùmero de llamadas al día")
plt.ylabel("Frecuencia")
plt.title("Histograma del número de llamadas por día")
plt.show()
