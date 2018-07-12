def pi_montecarlo(n, n_exp):
    import numpy as np
    pi_approximations = []
    for iteration in range(n_exp):
        success = 0
        x = np.random.uniform(0,1,n).tolist()
        y = np.random.uniform(0,1,n).tolist()
        for index in range(n):
            vector = np.power(x[index],2) + np.power(y[index],2)
            if vector < 1:
                success += 1
        pi_approximation = float(success)*4/n
        pi_approximations.append(pi_approximation)

    print(pi_approximations)
    print(np.mean(pi_approximations))
    

pi_montecarlo(10000, 200)
