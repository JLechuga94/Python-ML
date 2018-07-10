import pandas

data = pandas.read_csv("../../datasets/titanic/titanic3.csv")

def createDummies(dataset, var_name):
    print(dataset.head(10))
    dummy_variable = pandas.get_dummies(dataset[var_name], prefix = var_name)
    dataset = dataset.drop([var_name], axis = 1)
    dataset = pandas.concat([dataset, dummy_variable], axis = 1)
    print(dataset.head(10))

createDummies(data, "sex")
