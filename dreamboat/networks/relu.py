from .module import Module
import numpy as np


class ReLU(Module):
    '''
    relu linear rectify function,f(x) = max(x,0)
    '''
    def __init__(self):
        super().__init__()
        self.mask = None
        
    def forward(self, x):
        '''
        set a mask
        '''
        self.mask = x < 0
        x[self.mask] = 0
        return x
    
    def backward(self, dz):
        '''
        zeros gradients of mask
        '''
        dz[self.mask] = 0
        return dz
