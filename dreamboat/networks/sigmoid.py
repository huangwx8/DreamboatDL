from .module import Module
import numpy as np


class Sigmoid(Module):
    '''
    sigmoid activate function
    return values in [0,1]
    '''
    def __init__(self):
        super().__init__()
        self.out = None
        
    def forward(self, x):
        '''
        f(x) = 1/(1+np.exp(-x))
        '''
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backward(self, dz):
        '''
        f'(x) = f(x)*(1-f(x))
        '''
        return dz*self.out*(1-self.out)
