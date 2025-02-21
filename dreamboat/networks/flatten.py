from .module import Module
import numpy as np


class Flatten(Module):
    '''
    flatten images, (N,C,H,W)->(N,C*H*W)
    '''
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.output_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        out = x.reshape(x.shape[0],-1)
        self.output_shape = out.shape
        return out
    
    def backward(self, dz):
        assert self.output_shape == dz.shape
        dx = dz.reshape(self.input_shape)
        return dx
