# NNFS
A neural network implemented from scratch using only Numpy and some math.

# Design
Here I outline the design and some of the theory beind the most important functions of the network.

## Initialization
User defines layers as a list of `N` integers representing the size of each layer. For example, the list:
```
L = [10, 32, 32, 5]
```
would define a network with 10 inputs, two hidden layers with 32 neurons each, and a 5 neuron output layer.

The user also specifies a list of `N-1` activation functions for all but the input layer. For the network defined above, an example would be:
```
A = [sigmoid, sigmoid, relu]
```
Sigmoid and ReLU are defined by default, but more can be added as long as they follow the same format as sigmoid and ReLU.

## Forward Pass


## Backwards Pass


## Training


