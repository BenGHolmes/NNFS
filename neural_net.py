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
            
        weights, biases = init_NN_Glorot(layers, [squared_error, squared_error, squared_error], uniform=uniform)
        self._weights = weights
        self._biases = biases


    def forward(self, X):
        """Perform feed forward and get a prediction from the network.

        Args: 
            X (np.array): Input vector to the network with dimensions (batch_size, input_size, 1)

        Returns:
            a (list): List of activations for each layer
            z (list): List of weighted inputs for each layer
            y (np.array): Output vector of size (batch_size, output_size, 1)
        """
        y = X.copy()
        a = [X.copy()]
        z = []

        for l in range(self._L - 1):
            # Feed forward into weights and add biases
            y = np.einsum('spj, np -> snj', y, self._weights[l]) + self._biases[l]
            z.append(y.copy())        

            # Pass through activation function
            y = self._activations[l](y)
            a.append(y.copy())

        return a, z, y


    def backprop(self, X_batch, y_batch, a_batch, z_batch, out_batch, loss_func):
        """Return the gradients after running back propogation on a batch of inputs.
        
        Args:
            X_batch (ndarray): Array of shape (batch_size, input_size, 1)
            y_batch (ndarray): Array of shape (batch_size, output_size, 1)
            a_batch (list): List of activations for each layer.
            z_batch (list): List of weighted inputs for each layer.
            out_batch (ndarray): Predicted y from the network. Same shape as y_batch.
            loss_func (function): Function used to calculate loss.
       
        Returns:
            loss, (w_grads, b_grads): 
        """
        loss = loss_func(out_batch, y_batch)
        
        # Compute grads for last layer
        dC_da = loss_func(out_batch, y_batch, derivative=True)
        act_prime = self._activations[-1](z_batch[-1], derivative=True)
        delta_L = np.einsum('sij, sij -> sij', dC_da, act_prime)

        b_grads = [delta_L]
        w_grads = [np.einsum('sik,sjk -> sij', delta_L, a_batch[-2])]

        # Compute grads for other layers
        for l in range(2, self._L):
            act_prime = self._activations[-l](z_batch[-l], derivative=True)

            wT_delt = np.einsum('ij,sjk -> sik', self._weights[-l+1].T, b_grads[-1])
            delta_l = np.einsum('sij, sij -> sij', wT_delt, act_prime)
            b_grads.append(delta_l)
            w_grads.append(np.einsum('sik,sjk -> sij', delta_l, a_batch[-l-1]))

        # Reverse grad arrays and average for the batch
        avg_b_grads = [b.mean(axis=0) for b in reversed(b_grads)]
        avg_w_grads = [w.mean(axis=0) for w in reversed(w_grads)]

        return loss, (avg_w_grads, avg_b_grads)


    def train(self, X_train, y_train, X_test, y_test, epochs, loss_func, lr=0.01):
        """Perform back propogation and train the network

        Args:
            X_train (ndarray): Train input. Array of shape (n_batches, batch_size, input_size, 1)
            y_train (ndarray): Train targets. Array of shape (n_batches, batch_size, output_size, 1)
            X_test (ndarray): Test input. Array of shape (n_samples, input_size, 1)
            y_tet (ndarray): Test targets. Array of shape (n_samples, batch_size, output_size, 1)
            epochs (int): Number of epochs to train for.
            loss_func (function): Function used to calculate loss.
            lr (float): Learning rate.
        """
        for e in range(epochs):
            train_loss = 0
            for b in range(len(X_train)):
                a_batch, z_batch, out_batch = self.forward(X_train[b])
                batch_loss, (w_grads, b_grads) = self.backprop(X_train[b], y_train[b], a_batch, z_batch, out_batch, loss_func)
                train_loss = batch_loss

                for i, w in enumerate(self._weights):
                    w -= (w_grads[i] * lr)

                for i, b in enumerate(self._biases):
                    b -= (b_grads[i] * lr)

            if e % 100 == 99:
                _, _, test_pred = self.forward(X_test)
                test_loss = loss_func(test_pred, y_test)
                print(f'Epoch: {e+1}, Train loss: {train_loss.mean()}, Test loss: {test_loss.mean()}')


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
def squared_error(y, t, derivative=False):
    """Return squared error, or if derivative is true, the derivative of squared error"""
    if derivative:
        return 2*(y-t)
    else:
        return (y-t)**2


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

