import numpy as np
import matplotlib.pyplot as plt

data = np.random.randn(10000)

mu = 5.5
sd = 2.5
Z_10000 = np.random.randn(10000)
data2 = mu + sd*Z_10000 # Z = (X - mu)/sd --- N(0,1), X = mu + sd * Z
# Dataset that follows a uniform distribution dependen on abn variables
x = range(1,10001)
plt.figure(1)
plt.plot(x,data)
plt.figure(2)
plt.hist(data)
plt.figure(3)
plt.hist(data2)
plt.show()
