#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#graphing function
def graph(data, labels, noise):

    #set up path to directory, replace with path on your computer
    p = Path('/your/path/here/')

    #load data
    training_data = np.load(p / data)
    training_labels = np.load(p / labels)

    #set certain base values
    time = np.linspace(0, 0.01, num = 500)
    freq = np.fft.fftfreq(time.shape[-1])

    #find first actual wave
    i = 0

    while training_labels[i] != 1:
        i += 1

    print(training_labels[i])

    #make a figure of the wave
    plt.figure(figsize=(10.0, 6.0), constrained_layout=True)
    plt.subplot(2, 2, 1)
    plt.plot(time, training_data[i][0:500])
    plt.title("CW with {} Noise".format(noise))
    plt.xlabel("Time")

    #make a figure of the spectrum
    sp_true = np.abs(np.fft.fft(training_data[i][0:500]))

    plt.subplot(2, 2, 2)
    plt.plot(freq, sp_true.real)
    plt.title("CW with {} Noise Spectrum".format(noise))
    plt.xlabel("Frequency")

    #find first no wave
    j = 0

    while training_labels[j] != 0:
        j += 1

    #make a figure of the noise
    print(training_labels[j])
    plt.subplot(2, 2, 3)
    plt.plot(time, training_data[j][0:500])
    plt.title("No CW with {} Noise".format(noise))
    plt.xlabel("Time")

    #make a figure of the spectrum
    sp_false = np.abs(np.fft.fft(training_data[j][0:500]))

    plt.subplot(2, 2, 4)
    plt.plot(freq, sp_false.real)
    plt.title("No CW with {} Noise Spectrum".format(noise))
    plt.xlabel("Frequency")

    plt.savefig(p / "{}-noise-figure.png".format(noise))

#graph pair of graphs for each amount of noise

#0.5 noise data
graph("training-0.5-data.npy", "training-0.5-labels.npy", 0.5)

#1 noise data
graph("training-1-data.npy", "training-1-labels.npy", 1)

#2 noise data
graph("training-2-data.npy", "training-2-labels.npy", 2)

#3 noise data
graph("training-3-data.npy", "training-3-labels.npy", 3)


exit()