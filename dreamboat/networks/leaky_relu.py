from .module import Module
import numpy as np


class LeakyReLU(Module):
    '''
    relu linear rectify function with leak
    f(x) = x if x>0 else slope*x
    '''
    def __init__(self,slope):
        super().__init__()
        self.mask = None
        self.slope = slope
        
    def forward(self, x):
        self.mask = x < 0
        x[self.mask] *= self.slope
        return x
    
    def backward(self, dz):
        dz[self.mask] *= self.slope
        return dz
