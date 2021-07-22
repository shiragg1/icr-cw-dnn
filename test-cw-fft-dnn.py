#!/usr/bin/env python3

#TensorFlow and tf.keras
import tensorflow as tf

#helper libraries
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

p = Path('/your/path/here/')

def run_dnn(noise, num_runs):

    #set up path to directory, replace with path on your computer
    p = Path('/your/path/here/')

    #load data, replace number with correct number    
    training_data = np.load(p / "training-{}-data.npy".format(noise))
    training_labels = np.load(p / "training-{}-labels.npy".format(noise))
    testing_data = np.load(p / "testing-{}-data.npy".format(noise))
    testing_labels = np.load(p / "testing-{}-labels.npy".format(noise))

    class_names = ['no_wave', 'wave']

    accuracies = {}
    histories = {}

    for i in range(num_runs):
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
        history = model.fit(np.abs(np.fft.fft(training_data)), training_labels, epochs=15)

        #test the accuracy
        test_loss, test_acc = model.evaluate(np.abs(np.fft.fft(testing_data)), testing_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

        #examine the predictions
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(testing_data)

        print(predictions[0])
        print(np.argmax(predictions[0]))
        print(testing_labels[0])

        print(history.history.keys())

        accuracies[i] = test_acc
        histories[i] = history

        i += 1


    return {"histories" : histories,
            "accuracies" : accuracies}

dict_05 = run_dnn(0.5, 3)
np.save("0.5-accuracies.npy", dict_05["accuracies"])

dict_1 = run_dnn(1, 3)
np.save("1-accuracies.npy", dict_1["accuracies"])

dict_2 = run_dnn(2, 3)
np.save("2-accuracies.npy", dict_2["accuracies"])

dict_3 = run_dnn(3, 3)
np.save("3-accuracies.npy", dict_3["accuracies"])

dict_rand = run_dnn("rand", 1)
np.save("rand-accuracies.npy", dict_rand["accuracies"])

plt.plot(dict_05['histories'][0].history['accuracy'], label="0.5 Noise Data")
plt.plot(dict_1['histories'][0].history['accuracy'], label="1 Noise Data")
plt.plot(dict_2['histories'][0].history['accuracy'], label="2 Noise Data")
plt.plot(dict_3['histories'][0].history['accuracy'], label="3 Noise Data")
plt.plot(dict_rand['histories'][0].history['accuracy'], label="Random Noise Data")
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.savefig(p / "accuracy.png")