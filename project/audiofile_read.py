import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas
import re
import scipy
from sklearn import preprocessing

def load_sound_files(folder_path):
    files_names = sorted(os.listdir(folder_path))[1:] # Range is done due to DS_store file in index 0
    audio_files = [librosa.load(folder_path + file_name, duration=3.0) for file_name in files_names]
    files_names = [name.split(".")[0] for name in files_names]
    return audio_files, files_names

def fourier(audio_files):
    data_frame_x = []
    data_frame_y = []
    original_x = []
    original_y = []
    for index, audio_files in enumerate(audio_files):
        print("File index: " + str(index))
        print("Number of datapoints for audio file", len(data))
        data = audio_files[0]
        sampling_rate = audio_files[1]

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
        # print(len(t))
        # print(len(y))
        # y = y[:len(t)]
        data_frame_x.append(frq)
        data_frame_y.append(Y)
        # original_x.append(t)
        # original_y.append(y)

    return data_frame_x, data_frame_y, original_x, original_y

def create_dataframe(dataframe_data, label, file_names):
    for index, audio_data in enumerate(dataframe_data):
        data = pandas.DataFrame(audio_data)
        data = data.transpose()
        data.to_csv(global_path + "csv_{}/{}.csv".format(label, file_names[index]))
        print(data.shape)

def graph(x, y, t, original_y):
    fig, ax = plt.subplots(len(x), 2)
    for index in range(len(x)):
        # print(np.max(y[index]))
        ax[index][0].plot(t, original_y, 'b') # plotting the spectrum
        ax[index][1].plot(x[index], y[index],'r') # plotting the spectrum
        # ax[index][clear].set_ylim([0,1]) # plotting the spectrum
        ax[index][0].set_xlabel('Time')
        ax[index][0].set_ylabel('Amplitude')
        ax[index][1].set_xlabel('Freq (Hz)')
        ax[index][1].set_ylabel('|Y(freq)|')
    plt.show()

def concat_csv(folder_path):
    files_names = sorted(os.listdir(folder_path))[1:] # Range is done due to DS_store file in index 0
    csv_files = [pandas.read_csv(folder_path + file_name) for file_name in files_names]
    new_csv = pandas.concat(csv_files,index=False, ignore_index=True)
    new_csv.to_csv(folder_path + "COMPLETE.csv")
    return True

def cut_csv(folder_path):
    files_names = sorted(os.listdir(folder_path))[1:] # Range is done due to DS_store file in index 0
    csv_file = pandas.read_csv(folder_path + "global.csv")
    new_csv = csv_file.head(26)
    new_csv.to_csv(folder_path + "COMPLETE_CROP.csv")
    return True

# global_path = "../../../../../Downloads/CHALLENGE/"
global_path = "../../../../../../Downloads/datasets/CHALLENGE/"
# global_path = "audio_files/"
file_type = "normal"

patient_type = "abnormal"
database = "MITDATASET/"
# specific_database_name = "Atraining_{}/".format(patient_type)
specific_database_name = "training-a/{}/".format(patient_type)

global_path = "../../../../../../Downloads/datasets/"
folder_path = global_path + database + specific_database_name
new_path = global_path + database + "training-a/" + "divided_{}/".format(patient_type)



# print("Parsing: {} files".format(file_type))
audio_files, file_names = load_sound_files(global_path + "Atraining_{}/".format(file_type))
dataset_x, dataset_y, original_x, original_y = fourier(audio_files)
create_dataframe(dataset_y, file_type, file_names)
# graph(dataset_x,dataset_y, original_x, original_y)
concat_csv(global_path + "csv_{}/".format(file_type))
# cut_csv(global_path + "csv_{}/".format(file_type))
