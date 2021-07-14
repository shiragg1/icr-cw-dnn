#!/usr/bin/env python3
import numpy as np
from pathlib import Path

#set up path to directory, replace with path on your computer
p = Path('/home/shira/Documents/icr/icr-cw-dnn/')

#check 0.5 noise data

training_05_data = np.load(p / 'training-0.5-data.npy')
training_05_labels = np.load(p / 'training-0.5-labels.npy')

print('training data with 0.5 noise: ', training_05_data)
print('shape: ', training_05_data.shape)

print('labels for training data with 0.5 noise: ', training_05_labels)
print('shape: ', training_05_labels.shape)

testing_05_data = np.load(p / 'testing-0.5-data.npy')
testing_05_labels = np.load(p / 'testing-0.5-labels.npy')

print('testing data with 0.5 noise: ', testing_05_data)
print('shape: ', testing_05_data.shape)

print('labels for testing data with 0.5 noise: ', testing_05_labels)
print('shape: ', testing_05_labels.shape)

#check 1 noise data

training_1_data = np.load(p / 'training-1-data.npy')
training_1_labels = np.load(p / 'training-1-labels.npy')

print('training data with 1 noise: ', training_1_data)
print('shape: ', training_1_data.shape)

print('labels for training data with 1 noise: ', training_1_labels)
print('shape: ', training_1_labels.shape)

testing_1_data = np.load(p / 'testing-1-data.npy')
testing_1_labels = np.load(p /'testing-1-labels.npy')

print('testing data with 1 noise: ', testing_1_data)
print('shape: ', testing_1_data.shape)

print('labels for testing data with 1 noise: ', testing_1_labels)
print('shape: ', testing_1_labels.shape)

#check 2 noise data

training_2_data = np.load(p / 'training-2-data.npy')
training_2_labels = np.load(p / 'training-2-labels.npy')

print('training data with 2 noise: ', training_2_data)
print('shape: ', training_2_data.shape)

print('labels for training data with 2 noise: ', training_2_labels)
print('shape: ', training_2_labels.shape)

testing_2_data = np.load(p / 'testing-2-data.npy')
testing_2_labels = np.load(p /'testing-2-labels.npy')

print('testing data with 2 noise: ', testing_2_data)
print('shape: ', testing_2_data.shape)

print('labels for testing data with 2 noise: ', testing_2_labels)
print('shape: ', testing_2_labels.shape)

#check 3 noise data

training_3_data = np.load(p / 'training-3-data.npy')
training_3_labels = np.load(p / 'training-3-labels.npy')

print('training data with 3 noise: ', training_3_data)
print('shape: ', training_3_data.shape)

print('labels for training data with 3 noise: ', training_3_labels)
print('shape: ', training_3_labels.shape)

testing_3_data = np.load(p / 'testing-3-data.npy')
testing_3_labels = np.load(p /'testing-3-labels.npy')

print('testing data with 3 noise: ', testing_3_data)
print('shape: ', testing_3_data.shape)

print('labels for testing data with 3 noise: ', testing_3_labels)
print('shape: ', testing_3_labels.shape)