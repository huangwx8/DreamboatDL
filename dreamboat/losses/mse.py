import numpy as np


class MSELoss:
    '''
    mean square loss
    '''
    def __init__ (self):
        super().__init__()
        
        
    def __call__(self, x, y):
        '''
        loss = (x-y)^2
        dL/dx = 2(x-y)
        '''
        batch_size = np.prod(x.shape)
        loss = ((x-y)**2).mean()
        dx = 2*(x-y)
        return loss,dx/batch_size
