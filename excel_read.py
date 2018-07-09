import pandas

mainpath = "../datasets/"
filename = "titanic/titanic3.xls"

titanic3 = pandas.read_excel(mainpath + filename)
titanic3.to_csv(mainpath + "titanic/titanic_sugoi.csv")
titanic3.to_json(mainpath + "titanic/titanic_sugoi.json")
