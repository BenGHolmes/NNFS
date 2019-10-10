import numpy as np
import neural_net as nn

net = nn.Net([1, 3, 8, 3], [nn.sigmoid, nn.sigmoid, nn.sigmoid])
X = np.ones((1,4,1,1))
print(X)
y = net.forward(X[0])
print(y)