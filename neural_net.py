import numpy as np


class Net:
    def __init__(self, layers: list, activations: list):
        """Create the neural network and initialize the weights

        Args:
            layers (list): List of integers representing the number of neurons in each layer.
            activations (list): List of functions to be used as activations.
                len(activations) = len(layers) - 1
        """
        self._L = len(layers)
        self._layers = layers
        self._activations = activations

        self._weights = []
        self._bias = []
            
        for i in range(1, self._L):
            w = np.random.random((layers[i], layers[i-1]))  # N_out rows, and N_in columns
            b = np.random.random((layers[i], 1))
            self._weights.append(w)
            self._bias.append(b)


    def forward(self, X: np.array):
        """Perform feed forward and get a prediction from the network.

        Args: 
            X (np.array): Input vector to the network with dimensions (sample, input_size, 1)

        Returns:
            y (np.array): Output vector of size (sample, output_size, 1)
        """
        y = X.copy()
        for l in range(self._L - 1):
            # Feed forward into weights and add bias
            y = np.einsum('spj, np -> snj', y, self._weights[l]) + self._bias[l]

            # Pass through activation function
            y = self._activations[l](y)

        return y


    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float=0.01):
        """Perform back propogation and train the network

        Args:
            x_train (ndarray): Array of input vectors.
            y_train (ndarray): Array of true output vectors.
            epochs (int): Number of epochs to train for.
            lr (float): Learning rate.

        TODO:
            Me
        """
        pass            


########################
# Activation Functions #
########################
def sigmoid(x, derivative=False):
    """Return sigmoid(x), or if derivative is true, sigmoid'(x)"""
    f = 1/(1+np.exp(-x))  # sig(x)

    if derivative:              
        return f*(1-f)  # sig' = sig*(1-sig)
    else:                       
        return f  # return sig(x)
