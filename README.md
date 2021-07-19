# Implementing a Neural Network to Detect Simulated Gravitational Wave Data
Lexington Smith and Shira Goldhaber-Gordon, 2021 Institute for Computing in Research Project. Licensed under the GNU general Public License 3.0.

## Description
This deep neural network reads and analyzes data sets with an end goal of identifying a sine wave amidst noise. This sine wave is a model of a continuous gravitational wave meaning that it is possible for our neural net to use real LIGO data as well as what we generated. We used TensorFlow Keras to build this neural net which has performed well with current testing.  


## Installation and Instructions  
To clone the repository and access our code, run:

```bash
git clone git@github.com:shiragg1/icr-cw-dnn.git
```
To download the data files, run:

```bash
chmod +x load-data.sh
```

```bash
./load-data.sh
```
Alternatively to running these two commands -which will download all five data sets- you can just download one of the data sets directly from the .py file.
To make sure the data was downloaded correctly, run:

```bash
chmod +x data-check.py
```

```bash
./data-check.py
```
This will print out the arrays and their shapes. If you opted to download only one of the data sets you will have to edit the data-check.py file to only check for that data set.

Now to run the neural network. Set the correct path in cw-dnn.py and set the four data files to the file you wish to run. Then run:

```bash
chmod +x cw-dnn.py
```

```bash
./cw-dnn.py
```
We found higher accuracy in the neural network that took the fourier transforms of the data. We recomend running this DNN rather than the previous one. Set the correct path in cw-fft-dnn.py and set the four data files to the file you wish to run. Then run:

```bash
chmod +x cw-fft-dnn.py
```

```bash
./cw-fft-dnn.py
```
To create graphs displaying waves and their fourier transforms, set the correct path then run:

```bash
chmod +x graphs.py
```

```bash
./graphs.py
```

## Notes
*The numbers in the data set names refer to the standard deviation of noise within that given dataset.

*LIGO has not been able to detect continuous gravitational waves before; however, machine learning advances in neural networks like this one look promising for future directions.

*At the moment, the DNN accuracy is not extremely high for high noise data. However, if you have high computing power and can generate more data files than the default 2,000 the accuracy should improve substantially.

