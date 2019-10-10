import numpy as np


class Net:
    def __init__(self, layers, activations, uniform=False):
        """Create the neural network and initialize the weights

        Args:
            layers (list): List of integers representing the number of neurons in each layer.
            activations (list): List of functions to be used as activations.
                len(activations) = len(layers) - 1
            uniform (bool): Toggle whether or not to initialized network with uniform distribution.
        """
        self._L = len(layers)
        self._layers = layers
        self._activations = activations
            
        weights, biases = init_NN_Glorot(layers, activations, uniform=uniform)
        print(weights, biases)
        self._weights = weights
        self._biases = biases


    def forward(self, X):
        """Perform feed forward and get a prediction from the network.

        Args: 
            X (np.array): Input vector to the network with dimensions (batch_size, input_size, 1)

        Returns:
            y (np.array): Output vector of size (batch_size, output_size, 1)
        """
        y = X.copy()
        for l in range(self._L - 1):
            # Feed forward into weights and add biases
            y = np.einsum('spj, np -> snj', y, self._weights[l]) + self._biases[l]

            # Pass through activation function
            y = self._activations[l](y)

        return y


    def backprop(self, X_batch, y_batch, out_batch, loss_func):
        """Return the gradients after running back propogation on a batch of inputs.
        
        Args:
            X_batch (ndarray): Array of shape (batch_size, input_size, 1)
            y_batch (ndarray): Array of shape (batch_size, output_size, 1)
            out_batch (ndarray): Predicted y from the network. Same shape as y_batch.
            loss_func (function): Function used to calculate loss.
       
        Returns:
            loss, (w_grads, b_grads): 
        """
        loss = loss_func(out_batch, y_batch)
        



    def train(self, X_train, y_train, epochs, loss_func, lr=0.01):
        """Perform back propogation and train the network

        Args:
            X_train (ndarray): Array of shape (n_batches, batch_size, input_size, 1)
            y_train (ndarray): Array of shape (n_batches, batch_size, output_size, 1)
            epochs (int): Number of epochs to train for.
            loss_func (function): Function used to calculate loss.
            lr (float): Learning rate.

        TODO:
            Me
        """
        for e in range(epochs):
            for b in range(len(X_train)):
                out_batch = self.forward(X_train[b])
                loss, grads = self.backprop(X_train[b], y_train[b], out_batch, loss_func)


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


def tanh(x, derivative=False):
    """Return tanh(x), or if derivative is true, tanh'(x)"""
    x_safe = x + 1e-12  # Prevent overflow if x is zero
    f = (np.exp(x_safe)-np.exp(-x_safe))/(np.exp(x_safe)+np.exp(-x_safe))
    
    if derivative: 
        return 1-f**2
    else: 
        return f


def relu(x, derivative=False):
    """Return relu(x), or if derivative is true, relu'(x)"""
    if derivative:              
        return (x>0).astype(int)
    else:                       
        return np.maximum(x, 0)


##################
# Loss Functions #
##################
def rmse(y_pred, y_true):
    """Return the root, mean, squared error of y_pred relative to y_true"""
    return np.sqrt(np.mean((y_pred - y_true)**2))


############################
# Initialization Functions #
############################
def init_NN_Glorot(L, activations, uniform=False):
    """
    Initializer using the glorot initialization scheme
    
    Args:
        L (List): List containing number of neurons in each layer
        activations (List): Activation functions between each layer
        uniform (bool, optional): If true, use uniform distribution, else
            use a Gaussian distribution
    """
    weights = []
    biases = []
    
    for i in range(len(L) - 1):
        n_in = L[i]
        n_out = L[i+1]
        
        if activations[i].__name__ == 'tanh':
            if uniform:
                bound = (6. / (n_in + n_out)) ** 0.5 
                weights.append(np.random.uniform(low=-bound, high=bound, size=(n_out, n_in))) 
                biases.append(np.random.uniform(low=-bound, high=bound, size=(n_out, 1)))  
            else:
                std = (2. / (n_in + n_out)) ** 0.5
                weights.append(np.random.normal(loc=0.0, scale=std, size=(n_out, n_in))) 
                biases.append(np.random.normal(loc=0.0, scale=std, size=(n_out, 1))) 
                
        elif activations[i].__name__ == 'relu':
            if uniform:
                bound = (12. / (n_in + n_out)) ** 0.5
                weights.append(np.random.uniform(low=-bound, high=bound, size=(n_out, n_in))) 
                biases.append(np.random.uniform(low=-bound, high=bound, size=(n_out, 1)))  
            else:
                std = (4. / (n_in + n_out)) ** 0.5
                weights.append(np.random.normal(loc=0.0, scale=std, size=(n_out, n_in))) 
                biases.append(np.random.normal(loc=0.0, scale=std, size=(n_out, 1))) 
                    
        elif activations[i].__name__ == 'sigmoid':
            if uniform:
                bound = 4 * (6. / (n_in + n_out)) ** 0.5
                weights.append(np.random.uniform(low=-bound, high=bound, size=(n_out, n_in))) 
                biases.append(np.random.uniform(low=-bound, high=bound, size=(n_out, 1)))
            else:
                std = 4 * (2. / (n_in + n_out)) ** 0.5
                weights.append(np.random.normal(loc=0.0, scale=std, size=(n_out, n_in))) 
                biases.append(np.random.normal(loc=0.0, scale=std, size=(n_out, 1))) 
        else:
            # Use random normal distribution for linear activation
            weights.append(np.random.normal(loc=0.0, scale=1.0, size=(n_out, n_in))) 
            biases.append(np.random.normal(loc=0.0, scale=1.0, size=(n_out, 1)))  
    
    return (weights, biases)

