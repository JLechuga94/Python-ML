import os
import librosa
import numpy as np
# import matplotlib.pyplot as plt
import pandas
import re

global_path = "../../../../../../Downloads/datasets/CHALLENGE/"

def load_sound_files(folder_path):
    files_names = sorted(os.listdir(folder_path))[1:] # Range is done due to DS_store file in index 0
    audio_files = [librosa.load(folder_path + file_name) for file_name in files_names]
    files_names = [name.split(".")[0] for name in files_names]
    return audio_files, files_names

def fourier(audio_files):
    data_frame = []
    # fig, ax = plt.subplots(len(audio_files), 2, figsize=(20,8))
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
        div = 8000
        frq = frq[range(div)] # one side frequency range
        Y = np.fft.fft(y)/n # fft computing and normalization
        Y = Y[range(div)]

        data_frame.append(Y)

    #     ax[index][0].plot(t, y)
    #     ax[4][0].set_xlabel('Time')
    #     ax[4][0].set_ylabel('Amplitude')
    #     ax[index][1].plot(frq,abs(Y),'r') # plotting the spectrum
    #     ax[4][1].set_xlabel('Freq (Hz)')
    #     ax[4][1].set_ylabel('|Y(freq)|')
    # plt.show()
    return data_frame

def create_dataframe(dataframe_data, label, file_names):
    for index, audio_data in enumerate(dataframe_data):
        data = pandas.DataFrame({
            "Y_values_real": audio_data.real,
            "Y_values_imaginary": audio_data.imag
        })
        data.to_csv(global_path + "Aaudio_csv_{}/{}.csv".format(label, file_names[index]))

        # print(data.head())
        print(data.shape)

audio_files, file_names = load_sound_files(global_path + "Atraining_extrahls/")
dataset = fourier(audio_files)
create_dataframe(dataset, "extrahls", file_names)
