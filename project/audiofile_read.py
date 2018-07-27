import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas
import re
import scipy
from sklearn import preprocessing

global_path = "../../../../../../Downloads/datasets/CHALLENGE/"
# global_path = "audio_files/"

def load_sound_files(folder_path):
    files_names = sorted(os.listdir(folder_path))[1:] # Range is done due to DS_store file in index 0
    audio_files = [librosa.load(folder_path + file_name, duration=7.0) for file_name in files_names]
    files_names = [name.split(".")[0] for name in files_names]
    return audio_files, files_names

def fourier(audio_files):
    data_frame_x = []
    data_frame_y = []
    for index, audio_files in enumerate(audio_files):
        print(index)
        data = audio_files[0]
        sampling_rate = audio_files[1]

        Fs = sampling_rate;  # sampling rate
        Ts = 1.0/Fs; # sampling interval
        t = np.arange(0,len(data), 1) # time vector
        y = data

        n = len(y) # length of the signal
        k = np.arange(n)
        T = n/Fs
        frq = k/T # two sides frequency range
        div = int(n/2)
        frq = frq[range(div)] # one side frequency range
        i, = np.where(frq > 2500.0)
        frq = frq[range(i[0])]
        Y = np.fft.fft(y)
        for element in Y:
            # print(element)
            element = element/np.sum(Y)
            # print(element)
        print(Y)
        # min_max_scaler = preprocessing.MinMaxScaler()
        # Y = min_max_scaler.fit_transform(Y)

        # Y = [number/scipy.linalg.norm(Y) for number in Y]# fft computing and normalization
        Y = Y[range(i[0])]
        print(len(Y))
        data_frame_x.append(frq)
        data_frame_y.append(Y)


    return data_frame_x, data_frame_y

def create_dataframe(dataframe_data, label, file_names):
    for index, audio_data in enumerate(dataframe_data):
        data = pandas.DataFrame({
            "Y_values_real": audio_data.real,
            "Y_values_imaginary": audio_data.imag
        })
        data.to_csv(global_path + "Aaudio_csv_{}/{}.csv".format(label, file_names[index]))

        # print(data.head())
        print(data.shape)

def graph(x, y):
    fig, ax = plt.subplots(len(x), 1, squeeze=False)
    for index in range(len(x)):
        ax[index][0].plot(x[index],abs(y[index]),'r') # plotting the spectrum
        # ax[index][0].set_ylim([0,1]) # plotting the spectrum
        # ax[index][1].set_xlabel('Freq (Hz)')
        # ax[index][1].set_ylabel('|Y(freq)|')
    plt.show()




audio_files, file_names = load_sound_files(global_path + "Atraining_normal/")
dataset_x, dataset_y = fourier(audio_files)
# create_dataframe(dataset_y, "extrahls", file_names)
graph(dataset_x,dataset_y)
