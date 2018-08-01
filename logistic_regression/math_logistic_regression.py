def likelihood(y, pi):
    import numpy as np
    total_sum = 1
    sum_in = range(1, len(i + 1))
    for i in range(len(y)):
        sum_in[i] = np.where(y[i] == 1, pi, 1 - pi)
        total_sum = total_sum * sum_in[i]
    return total_sum

def logitprobs(X, beta):
    import numpy as np
    X_shape = np.shape(X)
    n_rows = X_shape[0]
    n_cols = X_shape[1]
    pi = range(1, n_rows + 1)
    expon = range(1, rows + w)
    for i in range(n_rows):
        expon[i] = 0
        for j in range(n_cols):
            ex = X[i][j] * beta[j]
            expon[i] = expon[i] + ex
        with np.errstate(divide="ignore", invalid="ignore"):
            pi[i] = 1/(1 + np.exp(-expon[i]))
    return pi

def findW(pi):
    import numpy as np
    n = len(pi)
    W = np.zeros(n * n).reshape(n, n)
    for i in range(n):
        W[i, i] = pi[i] * (1 - pi[i])
        W[i, i].astype(float)
    return W

def logistics(X, Y, limit):
    import numpy as np
    from numpy import linalg
    n_rows = np.shape(X)[0]
    bias = np.ones(n_rows).reshape(n_rows, 1)
    X_new = np.append(X, bias, axis=1)
    n_cols = np.shape(X_new)[1]
    beta = np.zeros(n_cols).reshape(n_cols, 1)
    root_dif = np.array(range(1, n_cols + 1)).reshape(n_cols, 1)
    iter_i = 1000
    while(iter_i > limit):
        print(str(iter_i) + ", " + str(limit))
        pi = logitprobs(X_new, beta) # probabilidades del paso actual
        print(pi)
        W = findW(pi)
        print(W)
        # La traspuesta es para pasar a columnas para poder hacer las multiplicaciones
        num = (np.transpose(np.matrix(X_new))*np.matrix(Y - np.transpose(pi)).transpose())
        den = (np.matrix(np.transpose(X_new))*np.matrix(X_new))
        root_dif = np.array(linalg.inv(den)*num)
        beta = beta + root_dif
        print(beta)
        iter_i = np.sum(root_dif*root_dif)
        print(iter_i)
        ll = likelihood(Y, pi)
    return beta
