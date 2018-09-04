from pydub import AudioSegment
from sklearn import preprocessing
import os
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas
import re
import scipy


path = "../../../../../../Downloads/Patients/"


files_names = sorted(os.listdir(path))
if files_names[0] == ".DS_Store":
    files_names = files_names[1:]

def cropp_file_5s():
    for file in files_names:
        t1 = 5000
        t2 = 10000
        file_id = file.split(".")[0]
        print("\nFile processed: " + file_id)
        original_audio = AudioSegment.from_wav(path + file)
        length = original_audio.duration_seconds
        print("Original audio length in seconds ", length)
        newAudio = original_audio[t1:t2]
        newAudio.export(path + file, format="wav")
        print("\nNew audio length in seconds", newAudio.duration_seconds)
        print("\nCropped audio file created ")
    return file

def FFT(file):
    audio_file = librosa.load(path + file, duration=5.0)
    data = audio_file[0]
    sampling_rate = audio_file[1]
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
    data_frame_x.append(frq)
    data_frame_y.append(Y)

    print("----------FFT processing terminated----------")
    return data_frame_x, data_frame_y

def create_dataframe(dataframe_data, label, file_names):
    print("\n-----------Beginning CSV file creation-----------")
    data = pandas.DataFrame(audio_data)
    data = data.transpose()
    data.to_csv(database_path + "{}.csv".format(label, file_names[index]), index=False)
    print("Rows | Columns", data.shape)
    counter += 1
    print(counter, "CSV files created")
    print("-----------CSV files creation finished-----------")
