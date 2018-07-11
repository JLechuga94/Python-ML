import numpy as np

n = 1000
pi_approximations = []
for iteration in range(100):
    success = 0
    x = np.random.uniform(0,1,n).tolist()
    y = np.random.uniform(0,1,n).tolist()
    for index in range(n):
        vector = np.power(x[index],2) + np.power(y[index],2)
        if vector < 1:
            success += 1
    probability = success/n
    pi_approximation = probability * 4
    pi_approximations.append(pi_approximation)

print(pi_approximations)
print(np.mean(pi_approximations))
