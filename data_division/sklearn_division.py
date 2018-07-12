from sklearn.model_selection import train_test_split
import pandas

data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")

train, test = train_test_split(data, test_size=0.2)

print(len(train))
print(len(test))
