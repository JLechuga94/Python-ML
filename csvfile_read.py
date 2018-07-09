import pandas as pd
import tabview as t
main_path = "../datasets"
data = pd.read_csv("../datasets/titanic/titanic3.csv")
t.view("../datasets/titanic/titanic3.csv")
