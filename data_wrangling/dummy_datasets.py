import pandas
import numpy as np

n = 100000
data = pandas.DataFrame(
    {
        'A': np.random.randn(n),
        'B': 1.5 + 2.5 * np.random.randn(n),
        'C': np.random.uniform(5, 32, n)
    }
)
data1 = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")

column_names = data1.columns.values.tolist()
a = len(column_names)

data2 = pandas.DataFrame(
    {
        'Column names': column_names,
        'A': np.random.randn(a),
        'B': 1.5 + 2.5 * np.random.randn(a),
        'C': np.random.uniform(5, 32, a)
    }, index=range(42, 42 + a)
    # Allows us to create values starting from a different index than 0 for
    # merging datasets'''
)

print(data1.head())
print(data2)
