import numpy as np


class SGD:
    '''
    SGD optimizer with momentum
    update equation:
    d <- beta*d+(1-beta)*g
    w <= w - lr*d
    '''
    def __init__(self,LEARNING_RATE=0.001,MOMENTUM=0.0):
        self.modules = []
        self.direct = []
        self.lr = LEARNING_RATE
        self.beta = MOMENTUM
        
    def add_module(self, module):
        self.modules.append(module)
        self.direct.append(dict())
        for key in module.state_dict:
            param = module.state_dict[key]['value']
            self.direct[-1][key] = np.zeros_like(param)
    
    def step(self):
        for i in range(len(self.modules)):
            for key in self.modules[i].state_dict:
                g = self.modules[i].state_dict[key]['grad']
                self.direct[i][key] *= self.beta
                self.direct[i][key] += (1-self.beta)*g
                self.modules[i].state_dict[key]['value'] -= self.lr*self.direct[i][key]
        return 'success'
