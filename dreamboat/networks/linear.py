from .module import Module
import numpy as np


class Linear(Module):
    '''
    The parameters of the Linear layer are trainable
    so you need to implement both the forward and backward propagation algorithms,
    as well as a process to update the parameters based on gradients.
    I choose the widely recognized and highly efficient Adam optimizer, and the parameters are initialized using Xavier initialization.
    '''
    def __init__(self,
                 in_features,
                 out_features):
        '''
        in_features: size of each input sample
        out_features: size of each output sample
        '''
        super().__init__()
        
        self._in_features = in_features
        self._out_features = out_features
        
        a = np.sqrt(5).item()
        bound = np.sqrt(6/((1+a**2)*in_features)).item()
        
        self.weight = (np.random.rand(in_features,out_features)-0.5)*2*bound
        self.bias  = np.zeros(out_features)
        
        self._dw = np.zeros_like(self.weight)
        self._db = np.zeros_like(self.bias)
        
        self.state_dict['weight'] = {'value':self.weight,'grad':self._dw}
        self.state_dict['bias'] = {'value':self.bias,'grad':self._db}
        
        self._latest_input = None

    def forward(self, x):
        '''
        calculate matrix multiply, and add a bias
        '''
        self._latest_input = x
        out =  np.dot(x,self.weight)+self.bias
        return out
    
    def backward(self, dz):
        '''
        dz: gradient in backpropagation
        call backward you will get gradient of weight and bias
        they are saved in self.state_dict['weight']['grad'] and
        self.state_dict['bias']['grad']
        '''
        batch_size,_ = self._latest_input.shape
        
        g_of_w = np.dot(self._latest_input.T,dz)
        g_of_b = np.sum(dz, axis = 0)
        
        self._dw += g_of_w
        self._db += g_of_b
        
        dx = np.dot(dz,self.weight.T)
        
        return dx
        
    def zero_grad(self):
        '''
        zeros the gradients of all parameters
        '''
        self._dw -= self._dw
        self._db -= self._db
