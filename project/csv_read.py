import pandas

data = pandas.read_csv("audios_csv/audio_file_0.csv")
print(data.head())
print(data.shape)
print(data.dtypes)
print(data["Y_values_imaginary"][1])
# for x in range(len(data["Y_values"][0])):
#     print(data["Y_values"][0][x])
