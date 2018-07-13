import pandas

# Obtenemos el primer archivo y lo usaremos como la base para concatenar
main_file = pandas.read_csv("../../datasets/distributed-data/001.csv")
total_length = len(main_file)
# Haciendo uso de formateo de STRING accedemos al nombre de cada archivo y lo
# concatenamos al archivo principal actualizandolo
for index in range(2, 333):
    print("File number {}".format(index))
    distributed_file = pandas.read_csv("../../datasets/distributed-data/{}.csv".format(format(index, "03d")))
    total_length += len(distributed_file)
    main_file = pandas.concat([main_file, distributed_file], axis=0)

# Al final se guarda el archivo completa como un .csv y se puede ordenar según
# el criterio deseado
print("The rows of final file corresponds to sum of rows for individual files: ")
print(total_length == main_file.shape[0])
main_file.to_csv("../../datasets/distributed-data/complete_data.csv")
