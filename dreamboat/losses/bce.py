import numpy as np


class BCELoss:
    '''
    binary cross entropy loss
    '''
    def __init__ (self):
        super().__init__()
        
        
    def __call__(self, x, y):
        '''
        loss = -ylog(x)-(1-y)log(1-x)
        dL/dx = 2(x-y)
        '''
        batch_size = np.prod(x.shape)
        loss = (-y*np.log(x)-(1-y)*np.log(1-x)).mean()
        dx = -y/x+(1-y)/(1-x)
        return loss,dx/batch_size
