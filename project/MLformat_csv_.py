from sklearn.model_selection import train_test_split
import pandas
import sklearn
import os


global_path = "../../../../../../Downloads/datasets/"

# database = "CHALLENGE/"
database = "MITDATASET/concat/"
database_path = global_path + database
def ML_processing(database, database_path):
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
    return 1

def concat_csv(database, database_path):
    print("Database selected:", database)
    print("\nLoading files from selected database...")
    files_names = sorted(os.listdir(database_path))
    if files_names[0] == ".DS_Store":
        files_names = files_names[1:]
    print("\nReading files...")
    csv_files = [pandas.read_csv(database_path + file_name) for file_name in files_names]
    print("\nConcatenation of files...")
    new_csv = pandas.concat(csv_files, axis=0)
    new_csv = sklearn.utils.shuffle(new_csv)
    new_csv.to_csv(database_path + "COMPLETE_ABCDF.csv", index=False)
    print("Processing finished")
    print("\n-----Head information of dataset-----")
    print(new_csv.head(5))
    print("\n-----Tail information of dataset----")
    print(new_csv.tail(5))
    print("\nDataset shape")
    print(new_csv.shape)
    return 1

# ML_processing(database, database_path)
concat_csv(database, database_path)
