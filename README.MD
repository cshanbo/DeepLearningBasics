This project contains Machine Learning basics wrote by myself

## About Input

1. The input of each model is a double matrix
2. If using batch GD, please set the input as the whole training data
2. If using SGD, please set the input as one training instance per time
4. If using SGD with mini-batch, please set the input as one batch per time

## Models

1. Logistic Regression
    1. Multi-class Logistic Regression (2-class also supported)
    2. Written in C++ 11
2. Hiddenlayer
    1. Hiddenlayer for mlp (feedforward neural network)
    2. could be used anywhere you want
    3. activate function includes:
        1. tanh (default)
        2. sigmoid (update fuc need to be changed)
        3. ReLU (update fuc need to be changed)
2. Multi-layer Perceptron (Feedforward NN)
    1. Single hidden layer MLP
    2. mini-batch supported
3. Retricted Boltzmann Machine
    1. First commit, bug fixing
    2. RBM done
4. Autoencoder
    1. done
5. CNN
    1. Convolutional layer
    2. need to construct a complete CNN
    3. need to complete the update function
        1. the most important step is to calculate the gradient of each layer
