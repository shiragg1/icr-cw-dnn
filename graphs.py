#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#set up path to directory, replace with path on your computer
p = Path('/home/shira/Documents/icr/icr-cw-dnn/')

#graph pair of graphs for each amount of noise

time = np.linspace(0, 0.1, num = 500)

#0.5 noise data

training_05_data = np.load(p / 'training-0.5-data.npy')
training_05_labels = np.load(p / 'training-0.5-labels.npy')

i = 0

while training_05_labels[i] != 1:
    i += 1

print(training_05_labels[i])
plt.figure(constrained_layout=True)
plt.subplot(2, 1, 1)
plt.plot(time, training_05_data[i][0:500])
plt.title("CW with 0.5 Noise")
plt.xlabel("Time")

j = 0

while training_05_labels[j] != 0:
    j += 1

print(training_05_labels[j])
plt.subplot(2, 1, 2)
plt.plot(time, training_05_data[j][0:500])
plt.title("No CW with 0.5 Noise")
plt.xlabel("Time")

plt.savefig(p / "0.5-noise-figure.png")

#1 noise data

training_1_data = np.load(p / 'training-1-data.npy')
training_1_labels = np.load(p / 'training-1-labels.npy')

i = 0

while training_1_labels[i] != 1:
    i += 1

print(training_1_labels[i])
plt.figure(constrained_layout=True)
plt.subplot(2, 1, 1)
plt.plot(time, training_1_data[i][0:500])
plt.title("CW with 1 Noise")
plt.xlabel("Time")

j = 0

while training_1_labels[j] != 0:
    j += 1

print(training_1_labels[j])
plt.subplot(2, 1, 2)
plt.plot(time, training_1_data[j][0:500])
plt.title("No CW with 1 Noise")
plt.xlabel("Time")

plt.savefig(p / "1-noise-figure.png")


#2 noise data

training_2_data = np.load(p / 'training-2-data.npy')
training_2_labels = np.load(p / 'training-2-labels.npy')

i = 0

while training_2_labels[i] != 1:
    i += 1

print(training_2_labels[i])
plt.figure(constrained_layout=True)
plt.subplot(2, 1, 1)
plt.plot(time, training_2_data[i][0:500])
plt.title("CW with 2 Noise")
plt.xlabel("Time")

j = 0

while training_2_labels[j] != 0:
    j += 1

print(training_2_labels[j])
plt.subplot(2, 1, 2)
plt.plot(time, training_2_data[j][0:500])
plt.title("No CW with 2 Noise")
plt.xlabel("Time")

plt.savefig(p / "2-noise-figure.png")


#3 noise data

training_3_data = np.load(p / 'training-3-data.npy')
training_3_labels = np.load(p / 'training-3-labels.npy')

i = 0

while training_3_labels[i] != 1:
    i += 1

print(training_3_labels[i])
plt.figure(constrained_layout=True)
plt.subplot(2, 1, 1)
plt.plot(time, training_3_data[i][0:500])
plt.title("CW with 3 Noise")
plt.xlabel("Time")

j = 0

while training_3_labels[j] != 0:
    j += 1

print(training_3_labels[j])
plt.subplot(2, 1, 2)
plt.plot(time, training_3_data[j][0:500])
plt.title("No CW with 3 Noise")
plt.xlabel("Time")

plt.savefig(p / "3-noise-figure.png")

#random noise data

training_rand_data = np.load(p / 'training-rand-data.npy')
training_rand_labels = np.load(p / 'training-rand-labels.npy')

i = 0

while training_rand_labels[i] != 1:
    i += 1

print(training_rand_labels[i])
plt.figure(constrained_layout=True)
plt.subplot(2, 1, 1)
plt.plot(time, training_rand_data[i][0:500])
plt.title("CW with Random Noise")
plt.xlabel("Time")

j = 0

while training_rand_labels[j] != 0:
    j += 1

print(training_rand_labels[j])
plt.subplot(2, 1, 2)
plt.plot(time, training_rand_data[j][0:500])
plt.title("No CW with Random Noise")
plt.xlabel("Time")

plt.savefig(p / "rand-noise-figure.png")

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