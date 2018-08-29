from sklearn.model_selection import train_test_split
import pandas
import sklearn


global_path = "../../../../../../Downloads/datasets/"

# database = "CHALLENGE/"
database = "MITDATASET/training-f/"
database_path = global_path + database

print("Database selected:", database)
print("\nProcessing of files...")
data_normal = pandas.read_csv(database_path + "csv_normal/COMPLETE.csv", index_col=0).drop(['Unnamed: 0.1'],axis=1)
data_normal["patient"] = "normal"
data_abnormal = pandas.read_csv(database_path + "csv_abnormal/COMPLETE.csv", index_col=0).drop(['Unnamed: 0.1'],axis=1)
data_abnormal["patient"] = "abnormal"
print("\nConcatenation, shuffling and saving of file...")
patient_data = pandas.concat([data_normal, data_abnormal], axis=0)
patient_data = sklearn.utils.shuffle(patient_data)
patient_data.to_csv(database_path + "COMPLETE_FORMATTED.csv", index=False)
print("Processing finished")
print("\n-----Head information of dataset-----")
print(patient_data.head(5))
print("\n-----Tail information of dataset----")
print(patient_data.tail(5))
print("\nDataset shape")
print(patient_data.shape)
