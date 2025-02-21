from .module import Module
import numpy as np


class Tanh(Module):
    '''
    tanh activate function
    return values in [-1,1]
    '''
    def __init__(self):
        super().__init__()
        self.out = None
        
    def forward(self, x):
        '''
        f(x) = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        '''
        self.out = np.tanh(x)
        return self.out
    
    def backward(self, dz):
        '''
        f'(x) = (1-f(x)*f(x))
        '''
        return dz*(1-self.out**2)
