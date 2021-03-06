#!/usr/bin/env python3

import numpy as np

#set up so the data can be replicated
np.random.seed(0)

#function generates points on a sin wave of given frequency and phase
def sin_wave(time, frequency, phase):

    return np.sin(2 * np.pi * frequency * time + phase)

#function generates random sin waves with added noise
def generate_true_data():
    time = np.linspace(0, 1, num = 50000)
    #frequencies based on the range of CWs
    frequency = np.random.uniform(low=20, high=1000)
    phase = np.random.uniform(low=0, high=np.pi*2)

    data = sin_wave(time, frequency, phase)
    #add gausian noise
    noise_amount = np.random.uniform(low=0.5, high=3)
    data_noisy = data + np.random.normal(0, noise_amount, data.shape)
    return data_noisy

#function generates random gaussian noise
def generate_false_data():
    data = np.zeros(50000, dtype=float)
    noise_amount = np.random.uniform(low=0.5, high=3)
    data_noisy = data + np.random.normal(0, noise_amount, data.shape)
    return data_noisy

training_data_array = np.array([generate_true_data()])
training_label_array = np.array([1])

#generate training data
for x in range(1600):
    data_type = np.random.randint(0,2)
    if data_type == 1:
        training_data_array = np.vstack((np.array([generate_true_data()]), training_data_array))
        training_label_array = np.vstack((np.array([1]), training_label_array))
    if data_type == 0:
        training_data_array = np.vstack((np.array([generate_false_data()]), training_data_array))
        training_label_array = np.vstack((np.array([0]), training_label_array))
    x+=1

#save training data
np.save("training-rand-data.npy", training_data_array)
np.save("training-rand-labels.npy", training_label_array)

testing_data_array = np.array([generate_true_data()])
testing_label_array = np.array([1])

#generate testing data
for x in range(400):
    data_type = np.random.randint(0,2)
    if data_type == 1:
        testing_data_array = np.vstack((np.array([generate_true_data()]), testing_data_array))
        testing_label_array = np.vstack((np.array([1]), testing_label_array))
    if data_type == 0:
        testing_data_array = np.vstack((np.array([generate_false_data()]), testing_data_array))
        testing_label_array = np.vstack((np.array([0]), testing_label_array))
    x+=1

#save testing data
np.save("testing-rand-data.npy", testing_data_array)
np.save("testing-rand-labels.npy", testing_label_array)
