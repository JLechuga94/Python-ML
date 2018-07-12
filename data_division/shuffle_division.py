import pandas
import sklearn

division_percentage = 0.75

data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")
data = sklearn.utils.shuffle(data)
print(data.head())

cut_id = int(division_percentage * len(data))
train_data = data[:cut_id]
test_data = data[cut_id:]

print(len(train_data))
print(len(test_data))
