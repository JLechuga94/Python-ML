import glob
import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

def load_sound_files(file):
    return librosa.load(file)

def plot_waves(data, sampling_rate):
    # plt.figure(figsize=(12, 4))
    librosa.display.waveplot(data, sr=sampling_rate)
    # plt.show()
    # i = 1
    # fig = plt.figure(figsize=(25,60), dpi = 900)
    # for n,f in zip(sound_names,raw_sounds):
    #     plt.subplot(10,1,i)
    #     librosa.display.waveplot(np.array(f),sr=22050)
    #     plt.title(n.title())
    #     i += 1
    # plt.suptitle('Figure 1: Waveplot',x=0.5, y=0.915,fontsize=18)
    # plt.show()
def fourier(data, sampling_rate):
    Fs = sampling_rate;  # sampling rate
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,len(data),1) # time vector
    y = data

    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    div = int(n/2)
    frq = frq[range(div)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(div)]
    print(len(Y))

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t, y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(frq,abs(Y),'r') # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.show()


data, sampling_rate = load_sound_files("201105021804.wav")
# print(data, sampling_rate)
# print(len(data))
# print(len(data)/(sampling_rate))
# plot_waves(data, sampling_rate)
fourier(data, sampling_rate)
