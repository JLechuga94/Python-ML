import numpy as np

# This allows to set a common starting point for random number generation
np.random.seed(2018)
print([np.random.random() for x in range(5)])
