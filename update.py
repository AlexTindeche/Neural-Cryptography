import numpy as np

def theta(t1, t2):
    if np.array_equal(t1, t2):
        return 1
    return 0
	

'''
    X: input
	W: weight matrix between input and hidden layer
	sigma: value of each hidden unit
	tau1: output of machine 1
	tau2: output of machine 2
	l: maximum weight value: {+l, ..., -3, -2, -1, 0, 1, 2, 3, ..., -l}
'''

def hebbian(W, X, sigma, tau1, tau2, l):
	for (i, j), _ in np.ndenumerate(W):
		W[i, j] += X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
		W[i, j] = np.clip(W[i, j] , -l, l) # bring the values outside the range of -l to l back to the range

def anti_hebbian(W, X, sigma, tau1, tau2, l):
	for (i, j), _ in np.ndenumerate(W):
		W[i, j] -= X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
		W[i, j] = np.clip(W[i, j], -l, l) # bring the values outside the range of -l to l back to the range

def random_walk(W, X, sigma, tau1, tau2, l):
	for (i, j), _ in np.ndenumerate(W):
		W[i, j] += X[i, j] * theta(sigma[i], tau1) * theta(tau1, tau2)
		W[i, j] = np.clip(W[i, j] , -l, l) # bring the values outside the range of -l to l back to the range