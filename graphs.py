#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#graphing function
def graph(data, labels, noise):

    #set up path to directory, replace with path on your computer
    p = Path('/home/shira/Documents/icr/icr-cw-dnn/')

    #load data
    training_data = np.load(p / data)
    training_labels = np.load(p / labels)

    #set certain base values
    time = np.linspace(0, 0.1, num = 500)
    freq = np.fft.fftfreq(time.shape[-1])

    #find first actual wave
    i = 0

    while training_labels[i] != 1:
        i += 1

    print(training_labels[i])

    #make a figure of the wave
    plt.figure(constrained_layout=True)
    plt.subplot(2, 2, 1)
    plt.plot(time, training_data[i][0:500])
    title = "CW with ", noise, " Noise"
    plt.title(title)
    plt.xlabel("Time")

    #make a figure of the spectrum
    sp_true = np.fft.fft(training_data[i][0:500])

    plt.subplot(2, 2, 2)
    plt.plot(freq, sp_true.real)
    title = "CW with ", noise, " Noise Spectrum"
    plt.title(title)
    plt.xlabel("Frequency")

    #find first no wave
    j = 0

    while training_labels[j] != 0:
        j += 1

    #make a figure of the noise
    print(training_labels[j])
    plt.subplot(2, 2, 3)
    plt.plot(time, training_data[j][0:500])
    title = "No CW with ", noise, " Noise"
    plt.title(title)
    plt.xlabel("Time")

    #make a figure of the spectrum
    sp_false = np.fft.fft(training_data[j][0:500])

    plt.subplot(2, 2, 4)
    plt.plot(freq, sp_false.real)
    title = "No CW with ", noise, " Noise Spectrum"
    plt.title(title)
    plt.xlabel("Frequency")

    plt.savefig(p / "{}-noise-figure.png".format(noise))

#graph pair of graphs for each amount of noise

#0.5 noise data
graph("training-0.5-data.npy", "training-0.5-labels.npy", "0.5")

#1 noise data
graph("training-1-data.npy", "training-1-labels.npy", "1")

#2 noise data
graph("training-2-data.npy", "training-2-labels.npy", "2")

#3 noise data
graph("training-3-data.npy", "training-3-labels.npy", "3")


exit()



# #grid of 15 dnn predictions

# def plot_value_array(i, predictions_array, true_label_array):
#   true_label = true_label_array[i]
#   plt.grid(False)
#   plt.xticks(range(10))
#   plt.yticks([])
#   thisplot = plt.bar(range(2), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)

#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')


# #plot the first 15 waves/no waves, their predicted labels, and the true labels
# #color correct predictions in blue and incorrect predictions in red
# time = np.linspace(0, 100, num = 500000)
# num_rows = 5
# num_cols = 3
# num_graphs = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_graphs):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plt.plot(time, testing_data[i])
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions[i], testing_labels)
# plt.tight_layout()
# plt.show()