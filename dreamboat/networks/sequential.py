from .module import Module
import numpy as np


class Sequential(Module):
    '''
    Sequentially connect many modules
    '''
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        
    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
    
    def backward(self, dz):
        for module in self.modules[::-1]:
            dz = module.backward(dz)
        return dz
                
    def zero_grad(self):
        for module in self.modules:
            module.zero_grad()
            
    def apply_optim(self, optimizer):
        for module in self.modules:
            optimizer.add_module(module)
