import numpy as np

from update import hebbian, anti_hebbian, random_walk

class TPM (object):
    '''
    A TPM is a three-layer neural network and the protocol for key agreement
    is based on the phenomenon of synchronization of two neural networks.

    A Three Parity Machine (TPM) is a type of non-linear Turing machine introduced as a computational model for 
    solving certain decision problems. It was proposed as a theoretical construct to demonstrate the computational 
    capabilities and limitations of non-linear machines.

    Parameters:
        k =  the number of hidden units
        n =  the number of units connected to the hidden units
        l =  maximum weight value: {+l, ..., -3, -2, -1, 0, 1, 2, 3, ..., -l}
        W =  weight matrix between input and hidden layer (k x n)    
    '''

    def __init__ (self, k, n, l):
        self.k = k
        self.n = n
        self.l = l
        self.W = np.random.randint(-l, l+1, (k, n))
        # print(self.W.shape)

    def output (self, X):
        '''
        Parameters:
            X = input (1 x n)
        '''
        
        sigma = np.sign(np.dot(X, self.W.T))
        tau = np.prod(sigma) # output of the TPM

        self.input = X
        self.sigma = sigma
        self.tau = tau

        return tau
    
    def __call__ (self, X):
        return self.output(X)
    
    def update (self, output2, learning_rule):
        '''
        Parameters:
            output2 = output of the second TPM which is used to synchronize the two TPMs
            learning_rule = {hebbian, anti_hebbian, random_walk}
        '''

        output1 = self.tau

        calls = {
            'hebbian': hebbian(self.W, self.input, self.sigma, output1, output2, self.l),
            'anti_hebbian': anti_hebbian(self.W, self.input, self.sigma, output1, output2, self.l),
            'random_walk': random_walk(self.W, self.input, self.sigma, output1, output2, self.l)
        }

        if output1 == output2:
            if learning_rule not in calls:
                raise ValueError('Invalid learning rule')
            calls[learning_rule]
        
        