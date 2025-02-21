from .module import Module
import numpy as np


class BatchNorm(Module):
    '''
    1-D batchnorm layer

    math:
        x_mean = (1/n)*sum(x,axis=1)
        x_cent = x-x_mean
        x_square = x_cent**2
        x_var = (1/n)*sum(x_square)
        x_std = sqrt(x_var)
        x_hat = x_cent/x_std
        y = w*x_hat+b
    '''
    def __init__(self, num_features):
        super().__init__()
        
        self.weight = np.ones(num_features)
        self.bias = np.zeros(num_features)
        
        self._dw = np.zeros_like(self.weight)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weight'] = {'value':self.weight,'grad':self._dw}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
        self.m = num_features
    
    def forward(self, x):
        n,m = x.shape
        assert m==self.m
        self.n = n
        
        self.x = x
        self.x_mean = np.mean(x,axis = 0)
        self.x_cent = x-self.x_mean
        self.x_square = self.x_cent**2
        self.x_var = np.mean(self.x_square,axis = 0)
        self.x_std = np.sqrt(self.x_var)
        self.x_frac_std = 1/self.x_std
        self.x_hat = self.x_cent*self.x_frac_std
        self.out = self.weight*self.x_hat+self.bias
        
        return self.out
    
    def backward(self, dout):
        # broadcast backward
        self._db += np.sum(dout, axis = 0)
        # dot product backward + broadcast backward
        self._dw += np.sum(self.x_hat*dout, axis = 0)
        # dot product backward
        self.dx_hat = self.weight*dout
        
        # dot product backward
        dx_cent1 = self.dx_hat*self.x_frac_std
        # dot product backward + broadcast backward
        dx_frac_std = np.sum(self.dx_hat*self.x_cent,axis=0)
        # divide backward
        dx_std = dx_frac_std*(-1/self.x_std**2)
        # sqrt backward
        dx_var = dx_std*(1/(2*self.x_std))
        # mean backward
        dx_square = np.tile(dx_var,[self.n,1])/self.n
        # square backward
        dx_cent2 = dx_square*(self.x_cent*2)
        # two sources
        dx_cent = dx_cent1+dx_cent2
        
        # add backward
        dx1 = dx_cent
        # minus backward + broadcast backward
        dx_mean = -np.sum(dx_cent,axis=0)
        # mean backward
        dx2 = np.tile(dx_mean,[self.n,1])/self.n
        # two sources
        dx = dx1+dx2
        
        return dx
    
    
    def zero_grad(self):
        self._db -= self._db
        self._dw -= self._dw
