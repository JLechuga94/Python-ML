import numpy as np
import random
import pandas

data = pandas.read_csv("../../datasets/customer-churn-model/Customer Churn Model.txt")

int_value = np.random.randint(1,100)
float_value = np.random.random()
print(int_value)
print(float_value)

def randint_list(n, a, b):
    return [np.random.randint(a,b) for x in range(n)]

def randfloat_list(n):
    return [np.random.random() for x in range(n)]

def randmultiple_list(n,a,b,m):
    return [random.randrange(a,b,m) for index in range(n)]

def shuffle(n):
    a = np.arange(n)
    print(a)
    np.random.shuffle(a)
    return a

def random_column(data):
    column_list = data.columns.values.tolist()
    return np.random.choice(column_list)

# print(randint_list(5,1,100))
# print(randfloat_list(5))
# print(randmultiple_list(10,0,100,7))
# print(shuffle(100))
print(random_column(data))
