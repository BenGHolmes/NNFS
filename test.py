import numpy as np
import neural_net as nn
import matplotlib.pyplot as plt

# Build fake data for y = 1 if (x > 5) else 0
# X in range [0,10]
X_train = np.random.random((100,64,1,1))*10
X_test = np.random.random((1000,1,1))*10

# Calculate y values for each X
y_train = (X_train > 5).astype(int)
y_test = (X_test > 5).astype(int)

# Build and train the network
L = [1,16,16,1]
A = [nn.sigmoid, nn.sigmoid, nn.sigmoid]
net = nn.Net(L,A)

net.train(X_train, y_train, X_test, y_test, 1000, nn.squared_error)

_,_,y_pred = net.forward(X_test)

plt.scatter(X_test.reshape(-1), y_pred.reshape(-1))
plt.show()