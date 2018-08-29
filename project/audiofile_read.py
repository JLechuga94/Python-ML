import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas
import re
import scipy
from sklearn import preprocessing

def load_sound_files(folder_path):
    print("\nLoading files from selected database...")
    files_names = sorted(os.listdir(folder_path))
    if files_names[0] == ".DS_Store":
        files_names = files_names[1:] # Range is done due to DS_store file in index 0
    audio_files = [librosa.load(folder_path + file_name, duration=3.0) for file_name in files_names]
    files_names = [name.split(".")[0] for name in files_names]
    print("Finished loading datafiles")
    return audio_files, files_names

def fourier(audio_files):
    data_frame_x = []
    data_frame_y = []
    original_x = []
    original_y = []
    print("\nNumber of files for processing ", len(audio_files))
    print("----------Beginning FFT processing----------")
    for index, audio_files in enumerate(audio_files):
        data = audio_files[0]
        sampling_rate = audio_files[1]
        print("\nFile index", index)
        print("Datapoints", len(data))

        Fs = sampling_rate;  # sampling rate
        Ts = 1.0/Fs; # sampling interval
        t = np.arange(0, 1, Ts) # time vector
        y = data

        n = len(y) # length of the signal
        k = np.arange(n)
        T = n/Fs
        frq = k/T # two sides frequency range
        div = int(n/2)
        frq = frq[range(div)] # one side frequency range
        i, = np.where(frq > 2500.0)
        frq = frq[range(i[0])]
        Y = np.fft.fft(y)/n
        Y = Y[range(div)] # one side frequency range
        Y = Y[range(i[0])]
        Y = [abs(number) for number in Y]
        Y = [element/np.max(Y) for element in Y]
        print("Size of data after normalization", len(Y))
        # print(len(t))
        # print(len(y))
        # y = y[:len(t)]
        data_frame_x.append(frq)
        data_frame_y.append(Y)
        # original_x.append(t)
        # original_y.append(y)
    print("----------FFT processing terminated----------")
    return data_frame_x, data_frame_y, original_x, original_y

def create_dataframe(dataframe_data, label, file_names):
    print("\n-----------Beginning CSV files creation-----------")
    counter = 0
    for index, audio_data in enumerate(dataframe_data):
        data = pandas.DataFrame(audio_data)
        data = data.transpose()
        data.to_csv(database_path + "csv_{}/{}.csv".format(label, file_names[index]))
        print("Rows | Columns", data.shape)
        counter += 1
    print(counter, "CSV files created")
    print("-----------CSV files creation finished-----------")

def concat_csv(folder_path):
    print("\nBeginning CSV files concatenation...")
    files_names = sorted(os.listdir(folder_path))[1:] # Range is done due to DS_store file in index 0
    csv_files = [pandas.read_csv(folder_path + file_name) for file_name in files_names]
    new_csv = pandas.concat(csv_files, ignore_index=True, sort=True)
    new_csv.to_csv(folder_path + "COMPLETE.csv")
    print("Finished CSV files concatenation")
    return True

def cut_csv(folder_path):
    files_names = sorted(os.listdir(folder_path))[1:] # Range is done due to DS_store file in index 0
    csv_file = pandas.read_csv(folder_path + "global.csv")
    new_csv = csv_file.head(26)
    new_csv.to_csv(folder_path + "COMPLETE_CROP.csv")
    return True

def graph(x, y, t, original_y):
    print(len(x))
    fig, ax = plt.subplots(len(x), 2)
    for index in range(len(x)):
        # print(x[index])
        # print(y[index])
        # print(np.max(y[index]))
        ax[index][0].plot(t, original_y, 'b') # plotting the spectrum
        ax[index][1].plot(x[index], y[index],'r') # plotting the spectrum
        # ax[index][clear].set_ylim([0,1]) # plotting the spectrum
        ax[index][0].set_xlabel('Time')
        ax[index][0].set_ylabel('Amplitude')
        ax[index][1].set_xlabel('Freq (Hz)')
        ax[index][1].set_ylabel('|Y(freq)|')
        plt.show()

# global_path = "../../../../../Downloads/"
global_path = "../../../../../../Downloads/datasets/"
# global_path = "audio_files/"
patient_type = "abnormal"

# database = "CHALLENGE/"
database = "MITDATASET/training-f/"
specific_database_name = "Atraining_{}/".format(patient_type)
# specific_database_name = "training-a/{}/".format(patient_type)
database_path = global_path + database




print("Database selected:", database)
print("Patient type:", patient_type)
audio_files, file_names = load_sound_files(database_path + "divided_{}/".format(patient_type))
dataset_x, dataset_y, original_x, original_y = fourier(audio_files)
create_dataframe(dataset_y, patient_type, file_names)
concat_csv(database_path + "csv_{}/".format(patient_type))
# graph(dataset_x,dataset_y, original_x, original_y)
# cut_csv(global_path + "csv_{}/".format(patient_type))
print("\n\n-----------Finished requested processing of database-----------")
