#!/usr/bin/env python3

#TensorFlow and tf.keras
import tensorflow as tf

#helper libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#set up path to directory, replace with path on your computer
p = Path('/home/shira/Documents/icr/icr-cw-dnn/')

#load data, replace number with correct number
training_data = np.load(p / "training-0.5-data.npy")
training_labels = np.load(p / "training-0.5-labels.npy")
testing_data = np.load(p / "testing-0.5-data.npy")
testing_labels = np.load(p / "testing-0.5-labels.npy")

class_names = ['no_wave', 'wave']

#allocate memory
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


#set up the layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(2)
])

#compile the model
model.compile(optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#train the model
model.fit(training_data, training_labels, epochs=10)

#test the accuracy
test_loss, test_acc = model.evaluate(testing_data, testing_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#examine the predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(testing_data)

print(predictions[0])
print(np.argmax(predictions[0]))
print(testing_labels[0])

#graph how well the dnn did
def plot_value_array(i, predictions_array, true_label_array):
  true_label = true_label_array[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(2), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#plot the first 15 waves/no waves, their predicted labels, and the true labels
#color correct predictions in blue and incorrect predictions in red
time = np.linspace(0, 100, num = 500000)
num_rows = 5
num_cols = 3
num_graphs = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_graphs):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plt.plot(time, testing_data[i])
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], testing_labels)
plt.tight_layout()
plt.show()